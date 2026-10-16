from typing import Any

from app.core.attack_store import AttackStore


def _compact_tactic(short_name: str, attack_store: AttackStore) -> dict[str, str] | None:
    tactic = attack_store.get_tactic(short_name)
    if tactic is None:
        return None
    return {
        "id": tactic["external_id"],
        "name": tactic["name"],
        "short_name": tactic["short_name"],
    }


def _compact_parent(technique_id: str, attack_store: AttackStore) -> dict[str, str] | None:
    if "." not in technique_id:
        return None
    parent = attack_store.get_technique(technique_id.split(".", 1)[0])
    if parent is None:
        return None
    return {"id": parent["external_id"], "name": parent["name"]}


def compact_techniques(technique_ids: list[str], attack_store: AttackStore) -> list[dict[str, Any]]:
    techniques: list[dict[str, Any]] = []
    for technique_id in technique_ids:
        technique = attack_store.get_technique(str(technique_id))
        if technique is None:
            continue
        tactics = [
            compact for short_name in technique.get("kill_chain_phases", [])
            if (compact := _compact_tactic(short_name, attack_store)) is not None
        ]
        item: dict[str, Any] = {
            "id": technique["external_id"],
            "name": technique["name"],
            "tactics": tactics,
            "is_subtechnique": bool(technique.get("is_subtechnique", False)),
        }
        parent = _compact_parent(technique["external_id"], attack_store)
        if parent is not None:
            item["parent"] = parent
        techniques.append(item)
    return techniques


def enrich_mitre(incident: dict[str, Any], attack_store: AttackStore) -> dict[str, Any]:
    techniques = compact_techniques(incident.get("mitre_technique_ids", []) or [], attack_store)
    tactics_by_id: dict[str, dict[str, str]] = {}
    for technique in techniques:
        for tactic in technique["tactics"]:
            tactics_by_id[tactic["id"]] = tactic
    return {"techniques": techniques, "tactics": list(tactics_by_id.values())}


def enrich_suggested_techniques(technique_ids: list[str], attack_store: AttackStore,
                                reason: str) -> list[dict[str, Any]]:
    result = compact_techniques(technique_ids, attack_store)
    for technique in result:
        technique.update({
            "disposition": "hypothesis",
            "verification": "unconfirmed",
            "reason": reason,
        })
    return result
