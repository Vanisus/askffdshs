from types import SimpleNamespace

from app.services.enrichment_pipeline import EnrichmentPipeline


class Store:
    def get_technique(self, tid):
        if tid not in {"T1071.004"}:
            return None
        return {"external_id": tid, "name": "DNS", "kill_chain_phases": ["command-and-control"], "is_subtechnique": True}
    def get_tactic(self, name):
        return {"external_id": "TA0011", "short_name": name, "name": "Command and Control"}


class Retriever:
    def search(self, query, top_k=20):
        return [{"doc_id": "1", "external_id": "T1071.004", "score": 0.8}]


class Reranker:
    def rerank(self, query, candidates, top_k=5):
        return [{"doc_id": "1", "external_id": "T1071.004", "rerank_score": 0.9}]


def container(debug=False):
    settings = SimpleNamespace(debug=debug, retrieval_top_k=20, rerank_top_k=5,
        rerank_score_threshold=0.35, rerank_relative_margin=0.65,
        rerank_min_gap=0.05, max_auto_techniques=3)
    return SimpleNamespace(settings=settings, attack_store=Store(), retriever=Retriever(),
        reranker=Reranker(), translator=None, result_cache=None)


def incident(**updates):
    value = {"ruleId": "completely-new-vendor-123", "ruleName": "DNS tunneling",
        "title": "DNS tunneling", "description": "large encoded DNS queries",
        "context": {}, "tags": [], "mitre_technique_ids": []}
    value.update(updates)
    return value


def test_public_output_is_compact_and_has_attack_metadata():
    result = EnrichmentPipeline(container()).run(incident())
    assert result.incident["mitre_technique_ids"] == ["T1071.004"]
    assert result.mitre == {
        "techniques": [{"id": "T1071.004", "name": "DNS", "tactics": [
            {"id": "TA0011", "name": "Command and Control", "short_name": "command-and-control"}],
            "is_subtechnique": True}],
        "tactics": [{"id": "TA0011", "name": "Command and Control", "short_name": "command-and-control"}],
        "decision": {"source": "automatic", "abstained": False, "reason": "selected", "confidence": 0.9},
    }


def test_debug_output_is_opt_in():
    result = EnrichmentPipeline(container(debug=True)).run(incident())
    assert result.mitre["debug"]["top_score"] == 0.9
    assert result.mitre["debug"]["retrieval_candidates"]
    assert "search_query" in result.mitre["debug"]


def test_low_confidence_abstains_with_public_reason():
    c = container()
    c.reranker.rerank = lambda *args, **kwargs: [{"doc_id": "1", "external_id": "T1071.004", "rerank_score": 0.1}]
    result = EnrichmentPipeline(c).run(incident(ruleName="unknown", title="unknown", description="suspicious process behavior"))
    assert result.mitre == {
        "techniques": [],
        "tactics": [],
        "decision": {"source": "automatic", "abstained": True, "reason": "below_threshold", "confidence": 0.1},
    }


def test_operational_health_event_does_not_map_to_attack():
    result = EnrichmentPipeline(container(debug=True)).run(incident(
        ruleId="rule-kes-001-ksn-unavailable", ruleName="", title="", description="",
        sourceId="ds-kaspersky-kes", context={"host_name": "KB43", "total_events": 1}))
    assert result.mitre["techniques"] == []
    assert result.mitre["decision"] == {
        "source": "automatic",
        "abstained": True,
        "reason": "no_observable_attack_behavior",
        "confidence": 0.0,
    }
    assert result.mitre["debug"]["reason"] == "no_observable_attack_behavior"


def test_source_provided_technique_is_explicit_in_public_decision():
    result = EnrichmentPipeline(container()).run(incident(mitre_technique_ids=["T1071.004"]))
    assert result.mitre["decision"] == {
        "source": "incident",
        "abstained": False,
        "reason": "source_provided",
        "confidence": 1.0,
    }
