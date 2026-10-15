# SIEM Enrichment Service

Production-oriented service that automatically maps SIEM incidents to MITRE ATT&CK techniques. It does not contain vendor-specific `ruleId` prefix tables, and `ruleId` is excluded from semantic search entirely.

## Evidence-first pipeline

1. Recursively extract schema-agnostic evidence from the incident: process, command line, parent, file, hash, URL/domain/IP, protocol, account, action and result.
2. Assess the activity as `adversary_behavior`, `suspicious_insufficient_evidence`, `not_attack_applicable`, `unsupported_input` or `analysis_failed`.
3. Finish early for non-attack and schema-poor events with a SOC-readable explanation and a list of missing evidence.
4. Translate natural-language fields containing Cyrillic; technical artifacts remain unchanged.
5. Retrieve ATT&CK candidates with BM25 and multilingual embeddings, merged by weighted RRF.
6. Rerank candidates with a cross-encoder and apply absolute score, relative score and top-gap gates.
7. Verify every surviving candidate against concrete incident evidence. Semantic similarity alone cannot create an ATT&CK mapping.
8. Resolve accepted technique and tactic metadata from the bundled ATT&CK dataset.
9. When an alert description clearly points to an ATT&CK candidate but evidence verification fails, expose it under `suggested_techniques` as an unconfirmed hypothesis. It is never copied to `incident.mitre_technique_ids`.

Specialized deterministic recognizers, such as the conservative CRL/OCSP detector, are retained only as fast paths. They produce the same general assessment contract and are not based on `ruleId`. Unknown certificate-looking URLs are not automatically trusted.

Explicit `mitre_technique_ids` supplied by the source incident always win and are marked as source-provided.

## Run

```bash
docker compose up --build
```

Endpoints:

- `POST /api/v1/enrich` — one incident.
- `POST /api/v1/enrich/batch` — bounded batch, `{ "incidents": [...] }`.
- `GET /health/live`, `GET /health/ready`.
- `GET /metrics` — Prometheus metrics.

## Load controls

- `SIEM_INFERENCE_CONCURRENCY=1` prevents concurrent model calls from exhausting one GPU.
- `SIEM_BATCH_MAX_SIZE=100` bounds batch fan-out.
- `SIEM_REQUEST_TIMEOUT_SECONDS=30` bounds request latency.
- `SIEM_RESULT_CACHE_SIZE=10000` and `SIEM_RESULT_CACHE_TTL_SECONDS=300` cache repeated correlations.

Run one process per GPU. Scale horizontally behind a load balancer; do not increase Uvicorn workers on one GPU because each worker loads its own model copies.

## Quality gates

Start with the defaults, then calibrate on labeled SOC incidents:

- `SIEM_RERANK_SCORE_THRESHOLD=0.35`
- `SIEM_RERANK_MIN_GAP=0.05`
- `SIEM_RERANK_RELATIVE_MARGIN=0.65`
- `SIEM_MAX_AUTO_TECHNIQUES=3`

Track precision@1, micro/macro F1, abstain rate, false-positive rate and p50/p95/p99 latency. Thresholds are deployment defaults, not universal truth; tune them on your own traffic.

## Tests

```bash
pip install pytest
pytest -q
```

## Response format

Every response contains an evidence-based `mitre.decision`:

```json
{
  "mitre": {
    "techniques": [],
    "tactics": [],
    "suggested_techniques": [
      {
        "id": "T1071.004",
        "name": "DNS",
        "tactics": [{"id": "TA0011", "name": "Command and Control", "short_name": "command-and-control"}],
        "is_subtechnique": true,
        "parent": {"id": "T1071", "name": "Application Layer Protocol"},
        "disposition": "hypothesis",
        "verification": "unconfirmed",
        "reason": "insufficient_evidence"
      }
    ],
    "decision": {
      "source": "automatic",
      "status": "suspicious_insufficient_evidence",
      "abstained": true,
      "reason": "insufficient_evidence",
      "confidence": 0.48,
      "summary": "Technical activity was observed, but its security intent cannot be established from this event alone.",
      "explanation": "ATT&CK mapping was withheld because semantic similarity alone is not proof of adversary behavior.",
      "missing_evidence": [
        "process lineage or command line",
        "behavior matching candidate ATT&CK semantics"
      ],
      "evidence": [
        {"path": "incident.events[0].fields.url", "kind": "url", "value": "http://example.test/a"}
      ],
      "context": {}
    }
  }
}
```

Decision statuses:

- `mapped` — candidate passed retrieval, score gates and evidence verification.
- `suspicious_insufficient_evidence` — suspicious or technical activity exists, but the technique is not proven.
- `not_attack_applicable` — operational or otherwise non-adversarial activity was identified.
- `ambiguous` — top candidates cannot be separated safely.
- `unsupported_input` — behavioral telemetry is missing.
- `analysis_failed` — the analyzer failed safely and no mapping was attempted.

Set `SIEM_DEBUG=true` only during diagnostics. It additionally returns normalized search text, retrieval/reranking candidates, per-technique verifier results, raw scores, cache state and processing time. Verbose internals remain absent from normal responses.


`mitre.techniques` contains only confirmed or source-provided mappings. `mitre.suggested_techniques` contains ATT&CK techniques/sub-techniques implied by the rule description or semantic candidates but not proven by telemetry. Public suggestions contain no model scores; scores and verifier details remain debug-only.
