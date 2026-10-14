from typing import Any

from app.core.attack_store import AttackStore


def enrich_mitre(incident: dict[str, Any], attack_store: AttackStore) -> dict[str, Any]:
    input_technique_ids = incident.get("mitre_technique_ids", []) or []

    matched_techniques: list[dict[str, Any]] = []
    matched_tactics_map: dict[str, dict[str, Any]] = {}
    unresolved_technique_ids: list[str] = []

    for technique_id in input_technique_ids:
        technique = attack_store.get_technique(technique_id)
        if technique is None:
            unresolved_technique_ids.append(technique_id)
            continue

        matched_techniques.append(
            {
                "id": technique["external_id"],
                "name": technique["name"],
                "tactics": technique["kill_chain_phases"],
                "is_subtechnique": technique.get("is_subtechnique", False),
            }
        )

        for short_name in technique.get("kill_chain_phases", []):
            tactic = attack_store.get_tactic(short_name)
            if tactic is None:
                continue
            matched_tactics_map[tactic["external_id"]] = {
                "id": tactic["external_id"],
                "short_name": tactic["short_name"],
                "name": tactic["name"],
            }

    return {
        "input_technique_ids": input_technique_ids,
        "matched_techniques": matched_techniques,
        "matched_tactics": list(matched_tactics_map.values()),
        "unresolved_technique_ids": unresolved_technique_ids,
    }