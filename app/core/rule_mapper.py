from typing import Any


_RULE_ID_HINTS: dict[str, tuple[str, ...]] = {
    "rule-ksmg-": ("T1499.002", "T1498"),
    "rule-ide-002": ("T1071.004", "T1048.003"),
    "rule-ide-": ("T1071.004",),
}


def suggest_mitre_mapping(incident: dict[str, Any]) -> dict[str, Any]:
    rule_id = str(incident.get("ruleId", "")).lower()

    matched: list[str] = []
    reasons: list[dict[str, str]] = []

    for prefix, technique_ids in _RULE_ID_HINTS.items():
        if rule_id.startswith(prefix):
            for tid in technique_ids:
                if tid not in matched:
                    matched.append(tid)
                    reasons.append(
                        {
                            "technique_id": tid,
                            "reason": f"ruleId prefix match: {prefix}",
                            "confidence": "medium",
                        }
                    )

    return {"mitre_technique_ids": matched, "mapping_reasons": reasons}
