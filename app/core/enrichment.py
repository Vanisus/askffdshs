from typing import Any

from app.core.attack_store import AttackStore


def enrich_mitre(incident: dict[str, Any], attack_store: AttackStore) -> dict[str, Any]:
    """Return compact ATT&CK metadata suitable for the public API."""
    technique_ids = incident.get("mitre_technique_ids", []) or []
    techniques: list[dict[str, Any]] = []
    tactics_by_id: dict[str, dict[str, Any]] = {}

    for technique_id in technique_ids:
        technique = attack_store.get_technique(str(technique_id))
        if technique is None:
            continue
        tactic_refs: list[dict[str, str]] = []
        for short_name in technique.get("kill_chain_phases", []):
            tactic = attack_store.get_tactic(short_name)
            if tactic is None:
                continue
            compact_tactic = {
                "id": tactic["external_id"],
                "name": tactic["name"],
                "short_name": tactic["short_name"],
            }
            tactics_by_id[compact_tactic["id"]] = compact_tactic
            tactic_refs.append(compact_tactic)
        techniques.append({
            "id": technique["external_id"],
            "name": technique["name"],
            "tactics": tactic_refs,
            "is_subtechnique": bool(technique.get("is_subtechnique", False)),
        })

    return {"techniques": techniques, "tactics": list(tactics_by_id.values())}
