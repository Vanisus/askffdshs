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

Every valid response includes a compact decision object, even when no technique is selected:

```json
{
  "mitre": {
    "techniques": [],
    "tactics": [],
    "decision": {
      "source": "automatic",
      "abstained": true,
      "reason": "below_threshold",
      "confidence": 0.12
    },
    "debug": null
  }
}
```

Possible automatic decision reasons are `matched`, `no_observable_attack_behavior`, `no_candidates`, `below_threshold` and `ambiguous_top_candidates`. Source-provided technique IDs use `source_provided`; invalid input uses `invalid_incident`.

### Request-scoped debug

Diagnostics are controlled per request rather than through an environment variable. For a single incident, add `"debug": true` to the top level of the request body:

```json
{
  "debug": true,
  "ruleId": "rule-kes-003-connection-blocked",
  "events": []
}
```

The service removes `debug` before incident validation and does not echo it into the normalized incident. A query parameter is also supported, which is convenient for a UI checkbox:

```text
POST /api/v1/enrich?debug=true
```

For a batch, place the flag next to `incidents` or use the same query parameter:

```json
{
  "debug": true,
  "incidents": [
    {"ruleId": "rule-1", "events": []},
    {"ruleId": "rule-2", "events": []}
  ]
}
```

When debug is enabled, `mitre.debug` includes the normalized search query, retrieval and reranking candidates, raw scores, cache state, selection diagnostics and processing time. Debug requests intentionally bypass result-cache reads so the candidate lists describe a fresh retrieval run. Without the flag, `mitre.debug` remains `null` and raw candidates are not exposed.
