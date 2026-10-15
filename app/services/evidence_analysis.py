from __future__ import annotations

import re
from typing import Any

from app.services.activity_classifier import detect_certificate_status_activity

ANALYZER_VERSION = "evidence-v1"

_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_HASH_RE = re.compile(r"\b(?:[0-9a-f]{32}|[0-9a-f]{40}|[0-9a-f]{64})\b", re.IGNORECASE)
_PROCESS_RE = re.compile(r"\b[\w.-]+\.(?:exe|dll|ps1|bat|cmd|sh|py)\b", re.IGNORECASE)
_ATTACK_RE = re.compile(
    r"\b(?:malware|ransom|phish|exploit|inject|credential|password|brute|"
    r"encoded|obfuscat|beacon|tunnel|exfil|lateral|persistence|payload|"
    r"reverse shell|command and control|c2|suspicious|unauthori[sz]ed|"
    r"вредонос\w*|фишинг\w*|эксплойт\w*|туннел\w*|подозритель\w*|"
    r"несанкционирован\w*)\b",
    re.IGNORECASE,
)
_OPERATIONAL_RE = re.compile(
    r"\b(?:unavailable|offline|timeout|health check|license|update failed|"
    r"service down|connection failed|certificate revocation|crl|ocsp|"
    r"недоступ\w*|сбой обновления|ошибка лицензии|проверка сертификат\w*)\b",
    re.IGNORECASE,
)
_KEY_KINDS = {
    "process": "process", "process_name": "process", "image": "process",
    "process_path": "process_path", "command": "command_line", "cmdline": "command_line",
    "command_line": "command_line", "powershell": "command_line", "parent": "parent_process",
    "parent_process": "parent_process", "url": "url", "uri": "url", "domain": "domain",
    "hostname": "domain", "query_name": "domain", "src_ip": "ip", "dst_ip": "ip",
    "source_ip": "ip", "destination_ip": "ip", "file": "file", "filename": "file",
    "path": "file_path", "sha256": "hash", "sha1": "hash", "md5": "hash",
    "hash": "hash", "user": "user", "user_name": "user", "account": "user",
    "protocol": "protocol", "proto": "protocol", "action": "action", "result": "result",
    "event_type": "event_type", "message": "message", "request": "request",
}
_NOISE_KEYS = {"event_id", "alert_id", "rule_id", "timestamp", "@timestamp", "created_at", "updated_at"}


class EvidenceExtractor:
    def extract(self, incident: dict[str, Any], limit: int = 32) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add(path: str, kind: str, value: Any) -> None:
            text = re.sub(r"\s+", " ", str(value)).strip()[:512]
            key = (kind, text.casefold())
            if text and key not in seen and len(items) < limit:
                seen.add(key)
                items.append({"path": path, "kind": kind, "value": text})

        for path, key, value in self._leaves(incident):
            normalized_key = re.sub(r"[^a-z0-9_@]", "_", key.lower()).strip("_")
            if normalized_key in _NOISE_KEYS:
                continue
            kind = _KEY_KINDS.get(normalized_key)
            if kind is not None and isinstance(value, (str, int, float)):
                add(path, kind, value)
            if not isinstance(value, str):
                continue
            for url in _URL_RE.findall(value):
                add(path, "url", url.rstrip(".,;:!?)]}\\"))
            for ip in _IPV4_RE.findall(value):
                if self._valid_ipv4(ip):
                    add(path, "ip", ip)
            for digest in _HASH_RE.findall(value):
                add(path, "hash", digest.lower())
            for process in _PROCESS_RE.findall(value):
                add(path, "process", process)
        return items

    def _leaves(self, value: Any, path: str = "incident", depth: int = 0):
        if depth > 8:
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                yield from self._leaves(nested, f"{path}.{key}", depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value[:100]):
                yield from self._leaves(nested, f"{path}[{index}]", depth + 1)
        else:
            yield path, path.rsplit(".", 1)[-1], value

    @staticmethod
    def _valid_ipv4(value: str) -> bool:
        try:
            return all(0 <= int(part) <= 255 for part in value.split("."))
        except ValueError:
            return False


class ActivityAnalyzer:
    def __init__(self, extractor: EvidenceExtractor | None = None) -> None:
        self.extractor = extractor or EvidenceExtractor()

    def analyze(self, incident: dict[str, Any]) -> dict[str, Any]:
        evidence = self.extractor.extract(incident)
        certificate = detect_certificate_status_activity(incident)
        if certificate is not None:
            context = certificate.get("context", {})
            return self._assessment(
                status="not_attack_applicable",
                reason="certificate_revocation_check",
                confidence=0.98,
                summary=f"Certificate revocation status request to {context.get('provider', 'a public CA')}.",
                explanation=str(certificate.get("explanation") or "Certificate-status traffic contains no attacker behavior by itself."),
                evidence=evidence,
                missing=["malicious process lineage", "payload execution", "adversary-controlled destination"],
                context=context,
            )

        natural = " ".join(
            str(incident.get(key) or "") for key in ("ruleName", "title", "description")
        )
        evidence_text = " ".join(item["value"] for item in evidence)
        combined = f"{natural} {evidence_text}".strip()
        attack_signal = bool(_ATTACK_RE.search(combined))
        operational_signal = bool(_OPERATIONAL_RE.search(combined))
        kinds = {item["kind"] for item in evidence}
        behavior_kinds = kinds.intersection({
            "process", "process_path", "parent_process", "command_line", "file", "file_path",
            "hash", "url", "domain", "ip", "protocol", "action", "result", "request",
        })

        if attack_signal:
            return self._assessment(
                status="adversary_behavior",
                reason="observable_attack_behavior",
                confidence=0.82,
                summary="The incident contains attack-relevant behavior or explicit suspicious indicators.",
                explanation="ATT&CK candidates may be evaluated, but each candidate must still be supported by the extracted evidence.",
                evidence=evidence,
                missing=self._missing_evidence(kinds),
            )
        if operational_signal:
            return self._assessment(
                status="not_attack_applicable",
                reason="benign_operational",
                confidence=0.86,
                summary="The incident describes operational, availability, update, or health-related activity.",
                explanation="No explicit adversary behavior was observed, so an ATT&CK technique was not assigned.",
                evidence=evidence,
                missing=["explicit suspicious action", "malicious process or command", "adversary objective"],
            )
        if behavior_kinds:
            return self._assessment(
                status="suspicious_insufficient_evidence",
                reason="insufficient_evidence",
                confidence=0.48,
                summary="Technical activity was observed, but its security intent cannot be established from this event alone.",
                explanation="The event is eligible for candidate retrieval, but mapping is accepted only when technique semantics match concrete evidence.",
                evidence=evidence,
                missing=self._missing_evidence(kinds),
            )
        return self._assessment(
            status="unsupported_input",
            reason="insufficient_evidence",
            confidence=0.2,
            summary="The incident does not contain enough behavioral telemetry for ATT&CK mapping.",
            explanation="Provide process, command-line, file, network, identity, action, or result evidence.",
            evidence=evidence,
            missing=["observable action", "actor or process", "target or object", "result"],
        )

    @staticmethod
    def search_fragment(assessment: dict[str, Any]) -> str:
        parts = []
        for item in assessment.get("evidence", [])[:16]:
            if item.get("kind") != "message":
                parts.append(f"{item.get('kind')}={item.get('value')}")
        return "; ".join(parts)

    @staticmethod
    def _missing_evidence(kinds: set[str]) -> list[str]:
        missing = []
        if not kinds.intersection({"process", "command_line", "parent_process"}):
            missing.append("process lineage or command line")
        if not kinds.intersection({"url", "domain", "ip", "protocol"}):
            missing.append("network destination or protocol")
        if not kinds.intersection({"action", "result", "event_type"}):
            missing.append("action and result")
        return missing

    @staticmethod
    def _assessment(*, status: str, reason: str, confidence: float, summary: str,
                    explanation: str, evidence: list[dict[str, str]], missing: list[str],
                    context: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "analyzer_version": ANALYZER_VERSION,
            "status": status,
            "reason": reason,
            "confidence": confidence,
            "summary": summary,
            "explanation": explanation,
            "evidence": evidence[:12],
            "missing_evidence": missing,
            "context": context or {},
        }
