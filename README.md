# SIEM Enrichment Service

Production-oriented service that automatically maps SIEM incidents to MITRE ATT&CK techniques. It does not contain vendor-specific `ruleId` prefix tables, and `ruleId` is excluded from semantic search entirely.

## Pipeline

1. Normalize `ruleName`, `title`, `description`, tags and useful technical context.
2. Translate only fields containing Cyrillic to English; already-English names and technical artifacts remain unchanged.
3. Retrieve candidates with BM25 and multilingual embeddings, merged by weighted RRF.
4. Rerank candidates with a cross-encoder.
5. Apply absolute score, relative score and top-gap gates. Ambiguous events return `abstained=true` instead of a fabricated mapping.
6. Resolve technique and tactic metadata from the bundled ATT&CK dataset.

Context is represented as canonical `field=value` evidence. Vendor prefixes in keys are stripped, duplicated values, IP addresses, UUIDs and timestamps are removed. Explicit `mitre_technique_ids` from the source incident always win and are marked as source-provided.

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

Normal responses always include a compact decision explaining whether an ATT&CK mapping was accepted:

```json
{
  "mitre": {
    "techniques": [{
      "id": "T1071.004",
      "name": "DNS",
      "tactics": [{"id": "TA0011", "name": "Command and Control", "short_name": "command-and-control"}],
      "is_subtechnique": true
    }],
    "tactics": [{"id": "TA0011", "name": "Command and Control", "short_name": "command-and-control"}],
    "decision": {
      "source": "automatic",
      "abstained": false,
      "reason": "selected",
      "confidence": 0.9
    }
  }
}
```

When no reliable mapping exists, `techniques` and `tactics` remain empty and `decision` explains the controlled refusal:

```json
{
  "mitre": {
    "techniques": [],
    "tactics": [],
    "decision": {
      "source": "automatic",
      "abstained": true,
      "reason": "no_observable_attack_behavior",
      "confidence": 0.0
    }
  }
}
```

Possible automatic reasons are `selected`, `no_observable_attack_behavior`, `no_candidates`, `below_threshold` and `ambiguous_top_candidates`. Explicit IDs supplied by the source use `source=incident` and `reason=source_provided`.

Set `SIEM_DEBUG=true` only during diagnostics. It additionally returns `mitre.debug` with the normalized search query, retrieval/reranking candidates, raw scores, cache state, selection details and processing time. Those potentially verbose internals stay absent from normal responses.
