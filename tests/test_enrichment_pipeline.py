from types import SimpleNamespace

from app.services.enrichment_pipeline import EnrichmentPipeline


class Store:
    def get_technique(self, tid):
        return {"external_id": tid, "name": tid, "kill_chain_phases": ["command-and-control"], "is_subtechnique": "." in tid}
    def get_tactic(self, name):
        return {"external_id": "TA0011", "short_name": name, "name": name}


class Retriever:
    def search(self, query, top_k=20):
        return [{"doc_id": "1", "external_id": "T1071.004", "rerank_score": 0.9}]


class Reranker:
    def rerank(self, query, candidates, top_k=5):
        return [{"doc_id": "1", "external_id": "T1071.004", "rerank_score": 0.9}]


def container(score=0.9):
    settings = SimpleNamespace(retrieval_top_k=20, rerank_top_k=5, rerank_score_threshold=0.35, rerank_relative_margin=0.65, rerank_min_gap=0.05, max_auto_techniques=3)
    c = SimpleNamespace(settings=settings, attack_store=Store(), retriever=Retriever(), reranker=Reranker(), translator=None, result_cache=None)
    return c


def test_automatic_mapping_does_not_use_rule_id_prefix():
    incident = {"ruleId": "completely-new-vendor-123", "ruleName": "DNS tunneling", "title": "DNS tunneling", "description": "large encoded DNS queries", "context": {}, "tags": [], "mitre_technique_ids": []}
    result = EnrichmentPipeline(container()).run(incident)
    assert result.mitre["suggested_technique_ids"] == ["T1071.004"]
    assert result.mitre["abstained"] is False


def test_low_confidence_abstains():
    c = container()
    c.reranker.rerank = lambda *args, **kwargs: [{"doc_id": "1", "external_id": "T1071.004", "rerank_score": 0.1}]
    incident = {"ruleId": "x", "ruleName": "unknown", "title": "unknown", "description": "unknown", "context": {}, "tags": [], "mitre_technique_ids": []}
    result = EnrichmentPipeline(c).run(incident)
    assert result.mitre["suggested_technique_ids"] == []
    assert result.mitre["abstained"] is True
