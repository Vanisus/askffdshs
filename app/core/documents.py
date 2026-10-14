from typing import Any


class MitreDocumentBuilder:
    def __init__(self, attack_store: Any) -> None:
        self.attack_store = attack_store

    def build(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []

        for technique_id, technique in self.attack_store.techniques_by_id.items():
            tactic_shortnames = technique.get("kill_chain_phases", []) or []

            search_parts = [
                technique.get("external_id", ""),
                technique.get("name", ""),
                technique.get("description", ""),
                " ".join(tactic_shortnames),
            ]

            documents.append(
                {
                    "doc_id": f"technique:{technique_id}",
                    "object_type": "technique",
                    "external_id": technique.get("external_id"),
                    "name": technique.get("name"),
                    "description": technique.get("description"),
                    "tactic_shortnames": tactic_shortnames,
                    "is_subtechnique": technique.get("is_subtechnique", False),
                    "search_text": " ".join(
                        str(p) for p in search_parts if p
                    ).strip(),
                }
            )

        for tactic_id, tactic in self.attack_store.tactics_by_id.items():
            search_parts = [
                tactic.get("external_id", ""),
                tactic.get("short_name", ""),
                tactic.get("name", ""),
                tactic.get("description", ""),
            ]

            documents.append(
                {
                    "doc_id": f"tactic:{tactic_id}",
                    "object_type": "tactic",
                    "external_id": tactic.get("external_id"),
                    "short_name": tactic.get("short_name"),
                    "name": tactic.get("name"),
                    "description": tactic.get("description"),
                    "search_text": " ".join(
                        str(p) for p in search_parts if p
                    ).strip(),
                }
            )

        return documents