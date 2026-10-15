from __future__ import annotations

import re
from typing import Any

_STOP = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "using", "used", "use",
    "adversary", "adversaries", "may", "can", "technique", "attack", "system",
})
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{2,}", re.IGNORECASE)


class TechniqueVerifier:
    def __init__(self, attack_store: Any) -> None:
        self.attack_store = attack_store

    def verify(self, candidate_ids: list[str], assessment: dict[str, Any], query: str,
               reranked: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        accepted: list[str] = []
        results: list[dict[str, Any]] = []
        observed = self._tokens(query + " " + " ".join(
            str(item.get("value") or "") for item in assessment.get("evidence", [])
        ))
        scores = {str(item.get("external_id", "")).upper(): float(item.get("rerank_score", 0.0))
                  for item in reranked}

        for technique_id in candidate_ids:
            technique = self.attack_store.get_technique(technique_id) or {}
            name_tokens = self._tokens(str(technique.get("name") or ""))
            description_tokens = self._tokens(str(technique.get("description") or ""))
            name_matches = sorted(observed.intersection(name_tokens))
            description_matches = sorted(observed.intersection(description_tokens))
            score = scores.get(technique_id, 0.0)
            has_semantic_evidence = bool(name_matches) or len(description_matches) >= 2
            accepted_candidate = (
                assessment.get("status") == "adversary_behavior" and has_semantic_evidence
            )
            if accepted_candidate:
                accepted.append(technique_id)
            results.append({
                "technique_id": technique_id,
                "verdict": "accepted" if accepted_candidate else "rejected",
                "confidence": round(max(0.0, min(1.0, score)), 6),
                "matched_terms": (name_matches + description_matches)[:12],
                "missing_evidence": [] if accepted_candidate else [
                    "observable behavior matching the ATT&CK technique semantics"
                ],
            })
        return accepted, results

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOP}
