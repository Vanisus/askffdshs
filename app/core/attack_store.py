import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AttackStore:
    def __init__(self, version: str, data_path: str | Path) -> None:
        self.version = version
        self.data_path = Path(data_path)

        self.loaded = False
        self.raw_bundle: dict[str, Any] | None = None

        self.techniques_by_id: dict[str, dict[str, Any]] = {}
        self.tactics_by_shortname: dict[str, dict[str, Any]] = {}
        self.tactics_by_id: dict[str, dict[str, Any]] = {}

    def load(self) -> bool:
        try:
            with self.data_path.open("r", encoding="utf-8") as f:
                bundle = json.load(f)

            objects = bundle.get("objects", [])
            if not isinstance(objects, list):
                raise ValueError("STIX bundle does not contain a valid 'objects' list")

            techniques_by_id: dict[str, dict[str, Any]] = {}
            tactics_by_shortname: dict[str, dict[str, Any]] = {}
            tactics_by_id: dict[str, dict[str, Any]] = {}

            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                if obj.get("revoked") is True:
                    continue
                if obj.get("x_mitre_deprecated") is True:
                    continue

                object_type = obj.get("type")
                external_id = self._extract_external_id(obj)

                if object_type == "attack-pattern":
                    if not external_id:
                        continue
                    techniques_by_id[external_id.upper()] = {
                        "stix_id": obj.get("id"),
                        "type": object_type,
                        "external_id": external_id.upper(),
                        "name": obj.get("name"),
                        "description": obj.get("description"),
                        "kill_chain_phases": self._extract_kill_chain_phases(obj),
                        "is_subtechnique": obj.get("x_mitre_is_subtechnique", False),
                    }
                elif object_type == "x-mitre-tactic":
                    short_name = obj.get("x_mitre_shortname")
                    if not external_id or not short_name:
                        continue
                    tactic = {
                        "stix_id": obj.get("id"),
                        "type": object_type,
                        "external_id": external_id.upper(),
                        "short_name": short_name.lower(),
                        "name": obj.get("name"),
                        "description": obj.get("description"),
                    }
                    tactics_by_shortname[short_name.lower()] = tactic
                    tactics_by_id[external_id.upper()] = tactic

            self.raw_bundle = bundle
            self.techniques_by_id = techniques_by_id
            self.tactics_by_shortname = tactics_by_shortname
            self.tactics_by_id = tactics_by_id
            self.loaded = True
            return True

        except Exception:
            logger.exception("AttackStore load error")
            self.raw_bundle = None
            self.techniques_by_id = {}
            self.tactics_by_shortname = {}
            self.tactics_by_id = {}
            self.loaded = False
            return False

    def get_technique(self, technique_id: str) -> dict[str, Any] | None:
        return self.techniques_by_id.get(technique_id.strip().upper())

    def get_tactic(self, short_name: str) -> dict[str, Any] | None:
        return self.tactics_by_shortname.get(short_name.strip().lower())

    def get_tactic_by_id(self, tactic_id: str) -> dict[str, Any] | None:
        return self.tactics_by_id.get(tactic_id.strip().upper())

    def health(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "version": self.version,
            "data_path": str(self.data_path),
            "techniques_count": len(self.techniques_by_id),
            "tactics_count": len(self.tactics_by_shortname),
        }

    @staticmethod
    def _extract_external_id(obj: dict[str, Any]) -> str | None:
        refs = obj.get("external_references", [])
        if not isinstance(refs, list):
            return None
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            if ref.get("source_name") == "mitre-attack":
                ext = ref.get("external_id")
                if isinstance(ext, str) and ext.strip():
                    return ext.strip()
        return None

    @staticmethod
    def _extract_kill_chain_phases(obj: dict[str, Any]) -> list[str]:
        result: list[str] = []
        phases = obj.get("kill_chain_phases", [])
        if not isinstance(phases, list):
            return result
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            if phase.get("kill_chain_name") != "mitre-attack":
                continue
            name = phase.get("phase_name")
            if isinstance(name, str) and name.strip():
                result.append(name.strip().lower())
        return result
