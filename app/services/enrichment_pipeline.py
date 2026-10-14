import logging
from dataclasses import dataclass
from typing import Any

from app.core import enrich_mitre, suggest_mitre_mapping
from app.core import query as incident_query
from app.services.container import AppContainer

logger = logging.getLogger(__name__)


_EXCLUDED_TACTICS: frozenset[str] = frozenset({
    "reconnaissance",
    "resource-development",
})


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
        attack_store = self.container.attack_store
        retriever = self.container.retriever
        reranker = self.container.reranker

        mapping = suggest_mitre_mapping(incident)

        query = incident_query.build_query_text(
            incident, self.container.translator
        )

        candidates = retriever.search(query, top_k=settings.retrieval_top_k)

        reranked: list[dict[str, Any]] = []
        if reranker is not None and candidates:
            reranked = reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=settings.rerank_top_k,
            )

        reranked = self._filter_pre_attack(reranked, attack_store)

        rag_ids = self._select_rag_ids(reranked, settings)

        final_ids = self._merge_ids(
            incident=incident,
            mapping=mapping,
            rag_ids=rag_ids,
        )

        # 8. Resolve + enrich
        mitre = enrich_mitre(incident, attack_store)
        mitre["mapping_reasons"] = mapping["mapping_reasons"]
        mitre["retrieval_candidates"] = candidates
        mitre["reranked_candidates"] = reranked
        mitre["suggested_technique_ids"] = final_ids
        mitre["search_query"] = query

        return EnrichmentResult(
            incident=incident,
            mitre=mitre,
            search_query=query,
        )

    @staticmethod
    def _filter_pre_attack(
        reranked: list[dict[str, Any]],
        attack_store: Any,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for candidate in reranked:
            tech = attack_store.get_technique(candidate["external_id"])
            if tech is None:
                out.append(candidate)
                continue

            tactics = set(tech.get("kill_chain_phases") or [])
            if tactics & _EXCLUDED_TACTICS:
                logger.debug(
                    "Filter %s: pre-attack tactic %s",
                    candidate["external_id"],
                    sorted(tactics),
                )
                continue

            out.append(candidate)

        return out

    @staticmethod
    def _select_rag_ids(
        reranked: list[dict[str, Any]],
        settings: Any,
    ) -> list[str]:
        if not reranked:
            return []

        top_score = float(reranked[0].get("rerank_score", 0.0))

        if top_score > 0:
            floor = max(
                settings.rerank_score_threshold,
                top_score * settings.rerank_relative_margin,
            )
        else:
            floor = settings.rerank_score_threshold

        return [
            c["external_id"]
            for c in reranked
            if float(c.get("rerank_score", 0.0)) >= floor
        ]

    @staticmethod
    def _merge_ids(
        incident: dict[str, Any],
        mapping: dict[str, Any],
        rag_ids: list[str],
    ) -> list[str]:
        explicit = list(incident.get("mitre_technique_ids") or [])
        if explicit:
            return explicit

        merged: list[str] = []
        for tid in list(mapping["mitre_technique_ids"]) + list(rag_ids):
            if tid and tid not in merged:
                merged.append(tid)

        incident["mitre_technique_ids"] = merged
        return merged
