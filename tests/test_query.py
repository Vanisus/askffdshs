from app.core.query import build_natural_parts, build_query_text, build_technical_text
from app.models import Incident


INCIDENT = {
    "alertId": "15a63310-286f-41e5-8166-2ad16af336cc",
    "ruleId": "rule-ide-002-long-subdomain",
    "ruleName": "Ideco DNS — Suspicious Long Subdomain (DNS Tunnel)",
    "severity": 3,
    "title": "Correlation: Ideco DNS — Suspicious Long Subdomain (DNS Tunnel)",
    "description": "DNS-запрос с меткой > 50 символов — возможный DNS-туннель",
    "triggeredAt": "2026-09-22T10:25:10.39093+00:00",
    "eventIds": ["4de441f5-e477-4f02-8a60-facc5f8751a6"],
    "context": {
        "p-ide-002.cs1": "SRV",
        "p-ide-002.src": "192.168.166",
        "p-ide-002.request": "_ntp._sctp.jogwi._msdcs.tdjame.",
        "p-ide-002.event_id": "4de441f5-e477-4f02-8a60-facc5f8751a6",
        "p-ide-002.client_ip": "192.168.166",
        "p-ide-002.@timestamp": "2026-09-22T10:25:10.3906385+00:00",
        "p-ide-002.query_name": "_ntp._sctp.jogwi._msdcs.tdjame.",
    },
    "tags": [],
}


class TranslatorSpy:
    def __init__(self):
        self.calls = []

    def translate_batch(self, texts):
        self.calls.append(texts)
        return ["DNS query with a label longer than 50 characters; possible DNS tunnel"]


def test_real_incident_query_is_clean_and_vendor_agnostic():
    normalized = Incident.model_validate(INCIDENT).model_dump(mode="json")
    translator = TranslatorSpy()
    query = build_query_text(normalized, translator)

    assert build_natural_parts(normalized) == [INCIDENT["ruleName"], INCIDENT["description"]]
    assert translator.calls == [[INCIDENT["description"]]]
    assert "rule-ide-002" not in query
    assert "Correlation:" not in query
    assert "192.168.166" not in query
    assert "4de441f5" not in query
    assert "2026-09-22" not in query
    assert query.count("_ntp._sctp.jogwi._msdcs.tdjame.") == 1
    assert "request=_ntp._sctp.jogwi._msdcs.tdjame." in query
    assert "DNS Tunnel" in query


def test_rule_id_is_never_technical_search_text():
    normalized = Incident.model_validate(INCIDENT).model_dump(mode="json")
    technical = build_technical_text(normalized)
    assert "rule-ide-002-long-subdomain" not in technical
