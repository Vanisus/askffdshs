from app.core.query import build_natural_parts, build_query_text, build_technical_text
from app.models import Incident


RAW_CORRELATION = {
    "ruleId": "rule-kes-001-ksn-unavailable",
    "sourceId": "ds-kaspersky-kes",
    "correlationKey": "agg:gn=115 отдел",
    "events": [
        {
            "eventId": "c55370ce-6941-47c5-9d96-5cb02c0387b8",
            "patternId": "p-kes-001",
            "timestamp": "2026-10-15T05:43:38.694146+00:00",
            "fields": {
                "hdn": "OTD115-01",
                "hip": "10.3.5.11",
                "gn": "115 отдел",
                "group": "115 отдел",
                "host_ip": "10.3.5.11",
                "host_name": "OTD115-01",
                "@timestamp": "2026-10-15T05:43:38.694146+00:00",
                "event_id": "c55370ce-6941-47c5-9d96-5cb02c0387b8",
            },
        },
        {
            "eventId": "229b86fd-a713-40c9-bcd2-98e3d7c88b15",
            "patternId": "p-kes-001",
            "timestamp": "2026-10-15T06:03:10.8214629+00:00",
            "fields": {
                "host_ip": "10.3.5.11",
                "host_name": "OTD115-01",
                "event_id": "229b86fd-a713-40c9-bcd2-98e3d7c88b15",
            },
        },
    ],
    "createdAt": "2026-10-15T05:43:38.6943172+00:00",
    "updatedAt": "2026-10-15T06:15:43.2010684+00:00",
    "aggregationState": {
        "results": {"gn=115 отдел.affected_hosts": 1},
        "totalEvents": 5,
    },
    "stateKey": "rule-kes-001-ksn-unavailable:agg:gn=115 отдел",
}


def test_raw_correlation_is_normalized_without_required_envelope():
    normalized = Incident.model_validate(RAW_CORRELATION).model_dump(mode="json")
    assert normalized["alertId"] is None
    assert normalized["ruleName"] == ""
    assert len(normalized["eventIds"]) == 2
    assert normalized["triggeredAt"].startswith("2026-10-15T06:15:43")
    assert normalized["context"]["host_name"] == "OTD115-01"
    assert normalized["context"]["total_events"] == 5
    assert normalized["context"]["affected_hosts"] == 1


def test_schema_poor_query_uses_generic_fallback_without_raw_id():
    normalized = Incident.model_validate(RAW_CORRELATION).model_dump(mode="json")
    query = build_query_text(normalized)
    assert "rule-kes-001-ksn-unavailable" not in query
    assert "ksn unavailable" in query
    assert "kaspersky" in query
    assert "10.3.5.11" not in query
    assert "c55370ce" not in query
    assert "host_name=OTD115-01" in query
    assert "total_events" not in query
    assert "affected_hosts" not in query
