import hashlib
import json
import logging
import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.core import enrich_mitre
from app.core import query as incident_query
from app.services.activity_classifier import detect_certificate_status_activity
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

    def run(self, incident: dict[str, Any]) -> EnrichmentResult:
        started = perf_counter()
        settings = self.container.settings
        explicit = self._normalized_explicit_ids(incident)

        # Source-provided ATT&CK IDs always win. Otherwise classify a narrow set
        # of explainable operational events before translation/retrieval so they
        # do not consume model capacity or receive a fabricated technique.
        activity_decision = None if explicit else detect_certificate_status_activity(incident)
        if activity_decision is not None:
            incident["mitre_technique_ids"] = []
            return self._result(incident, "", [], activity_decision, False, started)

        query = incident_query.build_query_text(incident, self.container.translator)
        cache_key = self._cache_key(query, explicit)
        cached = self.container.result_cache.get(cache_key) if self.container.result_cache else None
        if cached is not None:
            final_ids = list(cached["technique_ids"])
            incident["mitre_technique_ids"] = final_ids
            return self._result(incident, query, final_ids, cached.get("diagnostics", {}), True, started)

        candidates: list[dict[str, Any]] = []
        reranked: list[dict[str, Any]] = []
        if explicit:
            final_ids = explicit
            diagnostics = {"source": "incident", "abstained": False, "reason": "source_provided"}
        elif not incident_query.has_attack_behavior(incident, query):
            final_ids = []
            diagnostics = {"source": "automatic", "abstained": True, "reason": "no_observable_attack_behavior"}
        else:
            candidates = self.container.retriever.search(query, top_k=settings.retrieval_top_k)
            if self.container.reranker is not None and candidates:
                reranked = self.container.reranker.rerank(query, candidates, top_k=settings.rerank_top_k)
            reranked = self._filter_pre_attack(reranked)
            final_ids, diagnostics = self._select_automatic(reranked)

        incident["mitre_technique_ids"] = final_ids
        cache_payload = {"technique_ids": final_ids, "diagnostics": diagnostics}
        if self.container.result_cache:
            self.container.result_cache.put(cache_key, cache_payload)
        debug = {**diagnostics, "search_query": query, "retrieval_candidates": candidates,
                 "reranked_candidates": reranked, "cache_hit": False}
        return self._result(incident, query, final_ids, debug, False, started)

    def _result(self, incident: dict[str, Any], query: str, final_ids: list[str],
                diagnostics: dict[str, Any], cache_hit: bool, started: float) -> EnrichmentResult:
        mitre = enrich_mitre(incident, self.container.attack_store)
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        mitre["decision"] = self._public_decision(diagnostics, final_ids)
        if self.container.settings.debug:
            mitre["debug"] = {**diagnostics, "technique_ids": final_ids,
                              "cache_hit": cache_hit, "processing_ms": elapsed_ms}
        logger.info("MITRE enrichment techniques=%d abstained=%s reason=%s cache_hit=%s processing_ms=%.3f",
                    len(final_ids), mitre["decision"]["abstained"], mitre["decision"]["reason"],
                    cache_hit, elapsed_ms)
        return EnrichmentResult(incident=incident, mitre=mitre, search_query=query)

    @staticmethod
    def _public_decision(diagnostics: dict[str, Any], final_ids: list[str]) -> dict[str, Any]:
        source = str(diagnostics.get("source") or "automatic")
        abstained = bool(diagnostics.get("abstained", not bool(final_ids)))
        reason = diagnostics.get("reason")
        if not reason:
            reason = "selected" if final_ids else "no_candidates"

        if source == "incident" and final_ids:
            confidence = 1.0
        else:
            raw_score = diagnostics.get("top_score", 0.0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            if not math.isfinite(score):
                score = 0.0
            confidence = round(min(1.0, max(0.0, score)), 6)

        decision: dict[str, Any] = {
            "source": source,
            "abstained": abstained,
            "reason": str(reason),
            "confidence": confidence,
        }
        explanation = diagnostics.get("explanation")
        if explanation:
            decision["explanation"] = str(explanation)
        context = diagnostics.get("context")
        if isinstance(context, dict) and context:
            decision["context"] = context
        return decision

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
            return [], {"source": "automatic", "abstained": True, "reason": "no_candidates"}
        top = float(reranked[0].get("rerank_score", 0.0))
        second = float(reranked[1].get("rerank_score", 0.0)) if len(reranked) > 1 else 0.0
        if top < settings.rerank_score_threshold:
            return [], {"source": "automatic", "abstained": True, "reason": "below_threshold",
                        "top_score": top, "second_score": second}
        if len(reranked) > 1 and top - second < settings.rerank_min_gap:
            return [], {"source": "automatic", "abstained": True, "reason": "ambiguous_top_candidates",
                        "top_score": top, "second_score": second}
        floor = max(settings.rerank_score_threshold, top * settings.rerank_relative_margin)
        selected = [c for c in reranked if float(c.get("rerank_score", 0.0)) >= floor][:settings.max_auto_techniques]
        ids = list(dict.fromkeys(str(c["external_id"]).upper() for c in selected))
        return ids, {"source": "automatic", "abstained": not bool(ids), "reason": "selected" if ids else "no_candidates",
                     "top_score": top, "second_score": second, "selection_floor": floor}

    def _normalized_explicit_ids(self, incident: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for value in incident.get("mitre_technique_ids") or []:
            tid = str(value).strip().upper()
            if tid and self.container.attack_store.get_technique(tid) is not None and tid not in ids:
                ids.append(tid)
        return ids

    @staticmethod
    def _cache_key(query: str, explicit: list[str]) -> str:
        raw = json.dumps({"query": query, "explicit": explicit}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
