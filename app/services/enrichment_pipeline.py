import hashlib
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.core import enrich_mitre
from app.core import query as incident_query
from app.services.container import AppContainer

logger = logging.getLogger(__name__)
_EXCLUDED_TACTICS = frozenset({"reconnaissance", "resource-development"})


@dataclass(slots=True)
class EnrichmentResult:
    incident: dict[str, Any]
    mitre: dict[str, Any]
    search_query: str
    validation_errors: list[dict[str, Any]] | None = None


class EnrichmentPipeline:
    def __init__(self, container: AppContainer) -> None:
        self.container = container

    def run(self, incident: dict[str, Any], *, debug: bool = False) -> EnrichmentResult:
        started = perf_counter()
        settings = self.container.settings
        explicit = self._normalized_explicit_ids(incident)
        query = incident_query.build_query_text(incident, self.container.translator)
        cache_key = self._cache_key(query, explicit)

        cached = None
        if not debug and self.container.result_cache:
            cached = self.container.result_cache.get(cache_key)
        if cached is not None:
            final_ids = list(cached["technique_ids"])
            diagnostics = dict(cached.get("diagnostics", {}))
            incident["mitre_technique_ids"] = final_ids
            return self._result(
                incident,
                query,
                final_ids,
                diagnostics,
                debug_payload=None,
                cache_hit=True,
                started=started,
            )

        candidates: list[dict[str, Any]] = []
        reranked: list[dict[str, Any]] = []
        if explicit:
            final_ids = explicit
            diagnostics = {
                "source": "incident",
                "abstained": False,
                "reason": "source_provided",
                "confidence": 1.0,
            }
        elif not incident_query.has_attack_behavior(incident, query):
            final_ids = []
            diagnostics = {
                "source": "automatic",
                "abstained": True,
                "reason": "no_observable_attack_behavior",
                "confidence": 0.0,
            }
        else:
            candidates = self.container.retriever.search(query, top_k=settings.retrieval_top_k)
            if self.container.reranker is not None and candidates:
                reranked = self.container.reranker.rerank(query, candidates, top_k=settings.rerank_top_k)
            reranked = self._filter_pre_attack(reranked)
            final_ids, diagnostics = self._select_automatic(reranked)

        incident["mitre_technique_ids"] = final_ids
        cache_payload = {"technique_ids": final_ids, "diagnostics": diagnostics}
        if not debug and self.container.result_cache:
            self.container.result_cache.put(cache_key, cache_payload)

        debug_payload = None
        if debug:
            debug_payload = {
                **diagnostics,
                "search_query": query,
                "retrieval_candidates": candidates,
                "reranked_candidates": reranked,
                "cache_hit": False,
            }
        return self._result(
            incident,
            query,
            final_ids,
            diagnostics,
            debug_payload=debug_payload,
            cache_hit=False,
            started=started,
        )

    def _result(
        self,
        incident: dict[str, Any],
        query: str,
        final_ids: list[str],
        diagnostics: dict[str, Any],
        *,
        debug_payload: dict[str, Any] | None,
        cache_hit: bool,
        started: float,
    ) -> EnrichmentResult:
        mitre = enrich_mitre(incident, self.container.attack_store)
        mitre["decision"] = self._public_decision(diagnostics, final_ids)
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        if debug_payload is not None:
            mitre["debug"] = {
                **debug_payload,
                "technique_ids": final_ids,
                "cache_hit": cache_hit,
                "processing_ms": elapsed_ms,
            }
        logger.info(
            "MITRE enrichment techniques=%d abstained=%s reason=%s cache_hit=%s processing_ms=%.3f",
            len(final_ids),
            mitre["decision"]["abstained"],
            mitre["decision"]["reason"],
            cache_hit,
            elapsed_ms,
        )
        return EnrichmentResult(incident=incident, mitre=mitre, search_query=query)

    def _filter_pre_attack(self, reranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for candidate in reranked:
            tech = self.container.attack_store.get_technique(candidate["external_id"])
            tactics = set(tech.get("kill_chain_phases") or []) if tech else set()
            if tactics and tactics.issubset(_EXCLUDED_TACTICS):
                continue
            out.append(candidate)
        return out

    def _select_automatic(self, reranked: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
        settings = self.container.settings
        if not reranked:
            return [], {
                "source": "automatic",
                "abstained": True,
                "reason": "no_candidates",
                "confidence": 0.0,
            }
        top = float(reranked[0].get("rerank_score", 0.0))
        second = float(reranked[1].get("rerank_score", 0.0)) if len(reranked) > 1 else 0.0
        confidence = self._confidence(top)
        if top < settings.rerank_score_threshold:
            return [], {
                "source": "automatic",
                "abstained": True,
                "reason": "below_threshold",
                "confidence": confidence,
                "top_score": top,
                "second_score": second,
            }
        if len(reranked) > 1 and top - second < settings.rerank_min_gap:
            return [], {
                "source": "automatic",
                "abstained": True,
                "reason": "ambiguous_top_candidates",
                "confidence": confidence,
                "top_score": top,
                "second_score": second,
            }
        floor = max(settings.rerank_score_threshold, top * settings.rerank_relative_margin)
        selected = [
            candidate
            for candidate in reranked
            if float(candidate.get("rerank_score", 0.0)) >= floor
        ][: settings.max_auto_techniques]
        ids = list(dict.fromkeys(str(candidate["external_id"]).upper() for candidate in selected))
        return ids, {
            "source": "automatic",
            "abstained": not bool(ids),
            "reason": "matched" if ids else "no_candidates",
            "confidence": confidence,
            "top_score": top,
            "second_score": second,
            "selection_floor": floor,
        }

    def _normalized_explicit_ids(self, incident: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for value in incident.get("mitre_technique_ids") or []:
            tid = str(value).strip().upper()
            if tid and self.container.attack_store.get_technique(tid) is not None and tid not in ids:
                ids.append(tid)
        return ids

    @staticmethod
    def _public_decision(diagnostics: dict[str, Any], final_ids: list[str]) -> dict[str, Any]:
        return {
            "source": str(diagnostics.get("source") or "automatic"),
            "abstained": bool(diagnostics.get("abstained", not bool(final_ids))),
            "reason": str(diagnostics.get("reason") or ("matched" if final_ids else "no_candidates")),
            "confidence": EnrichmentPipeline._confidence(diagnostics.get("confidence", 0.0)),
        }

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return round(max(0.0, min(1.0, score)), 6)

    @staticmethod
    def _cache_key(query: str, explicit: list[str]) -> str:
        raw = json.dumps({"query": query, "explicit": explicit}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
