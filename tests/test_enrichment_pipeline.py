from types import SimpleNamespace

from app.services.enrichment_pipeline import EnrichmentPipeline


class Store:
    def get_technique(self, tid):
        if tid != "T1071.004":
            return None
        return {
            "external_id": tid, "name": "DNS",
            "description": "Adversaries may communicate using the DNS application layer protocol and DNS tunneling.",
            "kill_chain_phases": ["command-and-control"], "is_subtechnique": True,
        }

    def get_tactic(self, name):
        return {"external_id": "TA0011", "short_name": name, "name": "Command and Control"}


class Retriever:
    def search(self, query, top_k=20):
        return [{"doc_id": "1", "external_id": "T1071.004", "score": 0.8}]


class Reranker:
    def rerank(self, query, candidates, top_k=5):
        return [{"doc_id": "1", "external_id": "T1071.004", "rerank_score": 0.9}]


def container(debug=False):
    settings = SimpleNamespace(
        debug=debug, retrieval_top_k=20, rerank_top_k=5,
        rerank_score_threshold=0.35, rerank_relative_margin=0.65,
        rerank_min_gap=0.05, max_auto_techniques=3,
    )
    return SimpleNamespace(
        settings=settings, attack_store=Store(), retriever=Retriever(),
        reranker=Reranker(), translator=None, result_cache=None,
    )


def incident(**updates):
    value = {
        "ruleId": "completely-new-vendor-123", "ruleName": "DNS tunneling",
        "title": "DNS tunneling", "description": "large encoded DNS queries",
        "context": {}, "tags": [], "mitre_technique_ids": [],
    }
    value.update(updates)
    return value


def test_attack_event_is_mapped_only_after_evidence_verification():
    result = EnrichmentPipeline(container()).run(incident())
    assert result.incident["mitre_technique_ids"] == ["T1071.004"]
    assert result.mitre["decision"]["status"] == "mapped"
    assert result.mitre["decision"]["reason"] == "selected"
    assert result.mitre["decision"]["abstained"] is False
    assert result.mitre["decision"]["evidence"] == [] or isinstance(result.mitre["decision"]["evidence"], list)


def test_debug_output_contains_verifier_assessment():
    result = EnrichmentPipeline(container(debug=True)).run(incident())
    assert result.mitre["debug"]["top_score"] == 0.9
    assert result.mitre["debug"]["retrieval_candidates"]
    assert result.mitre["debug"]["technique_assessments"][0]["verdict"] == "accepted"


def test_low_confidence_abstains_with_public_reason():
    c = container()
    c.reranker.rerank = lambda *args, **kwargs: [
        {"doc_id": "1", "external_id": "T1071.004", "rerank_score": 0.1}
    ]
    result = EnrichmentPipeline(c).run(incident(
        ruleName="unknown suspicious process", title="unknown suspicious process",
        description="suspicious process behavior",
    ))
    decision = result.mitre["decision"]
    assert decision["status"] == "suspicious_insufficient_evidence"
    assert decision["reason"] == "below_threshold"
    assert decision["abstained"] is True


def test_operational_health_event_is_explained_without_retrieval():
    c = container(debug=True)
    c.retriever.search = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval must not run"))
    result = EnrichmentPipeline(c).run(incident(
        ruleId="rule-kes-001-ksn-unavailable", ruleName="", title="", description="KSN unavailable",
        sourceId="ds-kaspersky-kes", context={"host_name": "KB43", "total_events": 1},
    ))
    decision = result.mitre["decision"]
    assert decision["status"] == "not_attack_applicable"
    assert decision["reason"] == "benign_operational"
    assert decision["missing_evidence"]


def test_source_provided_technique_is_explicit_in_public_decision():
    result = EnrichmentPipeline(container()).run(incident(mitre_technique_ids=["T1071.004"]))
    assert result.mitre["decision"]["status"] == "mapped"
    assert result.mitre["decision"]["reason"] == "source_provided"
    assert result.mitre["decision"]["confidence"] == 1.0


def test_known_certificate_status_request_uses_specialized_fast_path():
    c = container()
    c.retriever.search = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval must not run"))
    result = EnrichmentPipeline(c).run(incident(
        ruleName="Connection blocked", title="Certificate status request blocked", description="",
        context={"message": "Server: http://r11.c.lencr.org/123.crl Result: connection blocked"},
    ))
    decision = result.mitre["decision"]
    assert decision["status"] == "not_attack_applicable"
    assert decision["reason"] == "certificate_revocation_check"
    assert decision["context"]["provider"] == "Let's Encrypt"
    assert decision["evidence"]


def test_nested_event_evidence_is_extracted_schema_agnostically():
    result = EnrichmentPipeline(container()).run(incident(
        ruleName="Suspicious DNS activity", title="DNS tunneling", description="encoded queries",
        context={}, events=[{"custom": {"image": "powershell.exe", "query_name": "tunnel.example"}}],
    ))
    kinds = {item["kind"] for item in result.mitre["decision"]["evidence"]}
    assert "process" in kinds
    assert "domain" in kinds


def test_unrelated_high_score_candidate_is_rejected_by_evidence_verifier():
    result = EnrichmentPipeline(container()).run(incident(
        ruleName="Suspicious payload download", title="Suspicious payload download",
        description="process downloaded executable content",
        context={"url": "http://unknown.example/payload.crl"},
    ))
    assert result.mitre["techniques"] == []
    assert result.mitre["decision"]["status"] == "suspicious_insufficient_evidence"
    assert result.mitre["decision"]["reason"] == "insufficient_evidence"
    assert "behavior matching candidate ATT&CK semantics" in result.mitre["decision"]["missing_evidence"]


def test_schema_poor_input_returns_actionable_missing_evidence():
    c = container()
    c.retriever.search = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval must not run"))
    result = EnrichmentPipeline(c).run({"ruleId": "vendor-123", "events": []})
    decision = result.mitre["decision"]
    assert decision["status"] == "unsupported_input"
    assert decision["reason"] == "insufficient_evidence"
    assert "observable action" in decision["missing_evidence"]
