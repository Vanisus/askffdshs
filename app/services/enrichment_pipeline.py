import hashlib
import json
import logging
import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.core import enrich_mitre
from app.core import query as incident_query
from app.services.container import AppContainer
from app.services.evidence_analysis import ANALYZER_VERSION, ActivityAnalyzer
from app.services.technique_verifier import TechniqueVerifier

logger = logging.getLogger(__name__)
_EXCLUDED_TACTICS = frozenset({"reconnaissance", "resource-development"})
_EARLY_ABSTAIN = frozenset({"not_attack_applicable", "unsupported_input"})


@dataclass(slots=True)
class EnrichmentResult:
    incident: dict[str, Any]
    mitre: dict[str, Any]
    search_query: str
    validation_errors: list[dict[str, Any]] | None = None


class EnrichmentPipeline:
    def __init__(self, container: AppContainer) -> None:
        self.container = container
        self.activity_analyzer = ActivityAnalyzer()
        self.technique_verifier = TechniqueVerifier(container.attack_store)

    def run(self, incident: dict[str, Any]) -> EnrichmentResult:
        started = perf_counter()
        settings = self.container.settings
        explicit = self._normalized_explicit_ids(incident)

        if explicit:
            assessment = None
        else:
            try:
                assessment = self.activity_analyzer.analyze(incident)
            except Exception:
                logger.exception("Universal activity analysis failed")
                assessment = {
                    "status": "analysis_failed", "reason": "analysis_failed", "confidence": 0.0,
                    "summary": "The incident could not be analyzed.",
                    "explanation": "The activity analyzer failed; ATT&CK mapping was not attempted.",
                    "evidence": [], "missing_evidence": ["successful activity analysis"], "context": {},
                }

        if assessment is not None and assessment["status"] in _EARLY_ABSTAIN.union({"analysis_failed"}):
            incident["mitre_technique_ids"] = []
            diagnostics = self._assessment_diagnostics(assessment)
            return self._result(incident, "", [], diagnostics, False, started)

        query = incident_query.build_query_text(incident, self.container.translator)
        if assessment is not None:
            fragment = self.activity_analyzer.search_fragment(assessment)
            if fragment:
                query = "\n".join(part for part in (query, f"Structured evidence: {fragment}") if part)
        cache_key = self._cache_key(query, explicit, assessment)
        cached = self.container.result_cache.get(cache_key) if self.container.result_cache else None
        if cached is not None:
            final_ids = list(cached["technique_ids"])
            incident["mitre_technique_ids"] = final_ids
            return self._result(incident, query, final_ids, cached.get("diagnostics", {}), True, started)

        candidates: list[dict[str, Any]] = []
        reranked: list[dict[str, Any]] = []
        technique_assessments: list[dict[str, Any]] = []
        if explicit:
            final_ids = explicit
            diagnostics = {
                "source": "incident", "status": "mapped", "abstained": False,
                "reason": "source_provided", "confidence": 1.0,
                "summary": "ATT&CK technique identifiers were supplied by the source incident.",
                "explanation": "Source-provided identifiers take precedence over automatic analysis.",
                "evidence": [], "missing_evidence": [],
            }
        else:
            candidates = self.container.retriever.search(query, top_k=settings.retrieval_top_k)
            if self.container.reranker is not None and candidates:
                reranked = self.container.reranker.rerank(query, candidates, top_k=settings.rerank_top_k)
            reranked = self._filter_pre_attack(reranked)
            proposed_ids, diagnostics = self._select_automatic(reranked)
            diagnostics.update(self._assessment_diagnostics(assessment or {}))
            diagnostics["reason"] = self._selection_reason(proposed_ids, reranked, diagnostics)
            if proposed_ids:
                final_ids, technique_assessments = self.technique_verifier.verify(
                    proposed_ids, assessment or {}, query, reranked
                )
                if final_ids:
                    diagnostics.update({
                        "status": "mapped", "abstained": False, "reason": "selected",
                        "summary": "Observed behavior supports the selected ATT&CK technique candidates.",
                        "explanation": "Candidates passed retrieval, reranking, confidence gates, and evidence verification.",
                    })
                else:
                    diagnostics.update({
                        "status": "suspicious_insufficient_evidence", "abstained": True,
                        "reason": "insufficient_evidence",
                        "summary": "Candidates were found, but none was supported by enough incident evidence.",
                        "explanation": "ATT&CK mapping was withheld because semantic similarity alone is not proof of adversary behavior.",
                    })
                    missing = list(diagnostics.get("missing_evidence") or [])
                    missing.append("behavior matching candidate ATT&CK semantics")
                    diagnostics["missing_evidence"] = list(dict.fromkeys(missing))
            else:
                final_ids = []
                diagnostics["status"] = (
                    "ambiguous" if diagnostics["reason"] == "ambiguous_top_candidates"
                    else "suspicious_insufficient_evidence"
                )
                diagnostics["abstained"] = True

        incident["mitre_technique_ids"] = final_ids
        diagnostics["technique_assessments"] = technique_assessments
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
        logger.info("MITRE enrichment techniques=%d status=%s reason=%s cache_hit=%s processing_ms=%.3f",
                    len(final_ids), mitre["decision"]["status"], mitre["decision"]["reason"],
                    cache_hit, elapsed_ms)
        return EnrichmentResult(incident=incident, mitre=mitre, search_query=query)

    @staticmethod
    def _assessment_diagnostics(assessment: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": "automatic",
            "status": assessment.get("status", "unsupported_input"),
            "abstained": assessment.get("status") != "adversary_behavior",
            "reason": assessment.get("reason", "insufficient_evidence"),
            "confidence": assessment.get("confidence", 0.0),
            "summary": assessment.get("summary", ""),
            "explanation": assessment.get("explanation", ""),
            "evidence": assessment.get("evidence", []),
            "missing_evidence": assessment.get("missing_evidence", []),
            "context": assessment.get("context", {}),
            "analyzer_version": assessment.get("analyzer_version", ANALYZER_VERSION),
        }

    @staticmethod
    def _selection_reason(ids: list[str], reranked: list[dict[str, Any]], diagnostics: dict[str, Any]) -> str:
        selection_reason = diagnostics.get("selection_reason")
        if selection_reason:
            return str(selection_reason)
        return "selected" if ids else ("no_candidates" if not reranked else "insufficient_evidence")

    @staticmethod
    def _public_decision(diagnostics: dict[str, Any], final_ids: list[str]) -> dict[str, Any]:
        source = str(diagnostics.get("source") or "automatic")
        status = str(diagnostics.get("status") or ("mapped" if final_ids else "suspicious_insufficient_evidence"))
        abstained = bool(diagnostics.get("abstained", not bool(final_ids)))
        reason = str(diagnostics.get("reason") or ("selected" if final_ids else "no_candidates"))
        raw_confidence = diagnostics.get("confidence", diagnostics.get("top_score", 0.0))
        try:
            score = float(raw_confidence)
        except (TypeError, ValueError):
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        decision: dict[str, Any] = {
            "source": source, "status": status, "abstained": abstained,
            "reason": reason, "confidence": round(min(1.0, max(0.0, score)), 6),
        }
        for key in ("summary", "explanation"):
            if diagnostics.get(key):
                decision[key] = str(diagnostics[key])
        for key in ("missing_evidence", "evidence"):
            if isinstance(diagnostics.get(key), list) and diagnostics[key]:
                decision[key] = diagnostics[key]
        if isinstance(diagnostics.get("context"), dict) and diagnostics["context"]:
            decision["context"] = diagnostics["context"]
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
            return [], {"selection_reason": "no_candidates"}
        top = float(reranked[0].get("rerank_score", 0.0))
        second = float(reranked[1].get("rerank_score", 0.0)) if len(reranked) > 1 else 0.0
        if top < settings.rerank_score_threshold:
            return [], {"selection_reason": "below_threshold", "top_score": top, "second_score": second}
        if len(reranked) > 1 and top - second < settings.rerank_min_gap:
            return [], {"selection_reason": "ambiguous_top_candidates", "top_score": top, "second_score": second}
        floor = max(settings.rerank_score_threshold, top * settings.rerank_relative_margin)
        selected = [c for c in reranked if float(c.get("rerank_score", 0.0)) >= floor][:settings.max_auto_techniques]
        ids = list(dict.fromkeys(str(c["external_id"]).upper() for c in selected))
        return ids, {"selection_reason": "selected" if ids else "no_candidates",
                     "top_score": top, "second_score": second, "selection_floor": floor}

    def _normalized_explicit_ids(self, incident: dict[str, Any]) -> list[str]:
        ids: list[str] = []
        for value in incident.get("mitre_technique_ids") or []:
            tid = str(value).strip().upper()
            if tid and self.container.attack_store.get_technique(tid) is not None and tid not in ids:
                ids.append(tid)
        return ids

    @staticmethod
    def _cache_key(query: str, explicit: list[str], assessment: dict[str, Any] | None) -> str:
        raw = json.dumps({
            "query": query, "explicit": explicit, "analyzer_version": ANALYZER_VERSION,
            "assessment_status": (assessment or {}).get("status"),
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
