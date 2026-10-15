import hashlib
import json
import logging
from dataclasses import dataclass
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

    def run(self, incident: dict[str, Any]) -> EnrichmentResult:
        settings = self.container.settings
        explicit = list(incident.get("mitre_technique_ids") or [])
        query = incident_query.build_query_text(incident, self.container.translator)
        cache_key = self._cache_key(query, explicit)
        cached = self.container.result_cache.get(cache_key) if self.container.result_cache else None
        if cached is not None:
            cached["cache_hit"] = True
            incident["mitre_technique_ids"] = list(cached["suggested_technique_ids"])
            return self._result(incident, query, cached)

        candidates = self.container.retriever.search(query, top_k=settings.retrieval_top_k)
        reranked = []
        if self.container.reranker is not None and candidates:
            reranked = self.container.reranker.rerank(query, candidates, top_k=settings.rerank_top_k)
        reranked = self._filter_pre_attack(reranked)

        if explicit:
            final_ids = explicit
            confidence = 1.0
            abstained = False
            reasons = [{"technique_id": tid, "reason": "provided by source incident", "confidence": 1.0} for tid in explicit]
        else:
            final_ids, confidence, abstained, reasons = self._select_automatic(reranked)
            incident["mitre_technique_ids"] = final_ids

        payload = {
            "mapping_reasons": reasons,
            "retrieval_candidates": candidates,
            "reranked_candidates": reranked,
            "suggested_technique_ids": final_ids,
            "search_query": query,
            "confidence": confidence,
            "abstained": abstained,
            "cache_hit": False,
        }
        if self.container.result_cache:
            self.container.result_cache.put(cache_key, payload)
        return self._result(incident, query, payload)

    def _result(self, incident: dict[str, Any], query: str, payload: dict[str, Any]) -> EnrichmentResult:
        mitre = enrich_mitre(incident, self.container.attack_store)
        mitre.update(payload)
        return EnrichmentResult(incident=incident, mitre=mitre, search_query=query)

    def _filter_pre_attack(self, reranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for candidate in reranked:
            tech = self.container.attack_store.get_technique(candidate["external_id"])
            tactics = set(tech.get("kill_chain_phases") or []) if tech else set()
            if not tactics.intersection(_EXCLUDED_TACTICS):
                out.append(candidate)
        return out

    def _select_automatic(self, reranked: list[dict[str, Any]]) -> tuple[list[str], float, bool, list[dict[str, Any]]]:
        settings = self.container.settings
        if not reranked:
            return [], 0.0, True, []
        top = float(reranked[0].get("rerank_score", 0.0))
        second = float(reranked[1].get("rerank_score", 0.0)) if len(reranked) > 1 else 0.0
        if top < settings.rerank_score_threshold or (len(reranked) > 1 and top - second < settings.rerank_min_gap):
            return [], max(0.0, min(1.0, top)), True, []
        floor = max(settings.rerank_score_threshold, top * settings.rerank_relative_margin)
        selected = [c for c in reranked if float(c.get("rerank_score", 0.0)) >= floor][: settings.max_auto_techniques]
        ids = [c["external_id"] for c in selected]
        reasons = [{
            "technique_id": c["external_id"],
            "reason": "automatic hybrid retrieval and cross-encoder reranking",
            "confidence": round(max(0.0, min(1.0, float(c.get("rerank_score", 0.0)))), 6),
        } for c in selected]
        return ids, round(max(0.0, min(1.0, top)), 6), not bool(ids), reasons

    @staticmethod
    def _cache_key(query: str, explicit: list[str]) -> str:
        raw = json.dumps({"query": query, "explicit": explicit}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
