import logging
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TranslatorProtocol(Protocol):
    def translate(self, text: str) -> str: ...
    def translate_batch(self, texts: list[str]) -> list[str]: ...


_CONTEXT_ALLOWLIST = frozenset({
    "message", "request", "query", "query_name", "hostname", "host_name",
    "mail_host", "host", "domain", "url", "uri", "path", "file", "process",
    "process_name", "process_path", "command", "image", "parent", "parent_process",
    "powershell", "registry", "registry_key", "task", "service", "user_agent",
    "dns", "cmdline", "command_line", "subject", "account", "sha", "hash",
    "md5", "sha1", "sha256", "filename", "mail", "src", "dst", "src_port",
    "dst_port", "port", "protocol", "proto", "user", "user_name", "record_type",
    "event_type", "action", "result", "object", "target", "network_direction",
})
_CONTEXT_DENYLIST = frozenset({
    "event_id", "alert_id", "rule_id", "timestamp", "@timestamp", "criticality",
    "owner", "department", "group", "gn", "conn_count", "count", "severity",
    "total_events", "affected_hosts", "host_ip", "hip", "hdn",
})
_IP_RE = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){1,3}(?![\w.])")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:?\d{2})?\b", re.IGNORECASE)
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_GENERIC_ID_TOKENS = frozenset({"rule", "alert", "correlation", "event", "source", "ds", "p"})
_OPERATIONAL_ONLY_RE = re.compile(
    r"\b(unavailable|not available|offline|disconnected|connection failed|timeout|"
    r"license|database error|service (?:is )?down|update failed|out of date|"
    r"недоступ\w*|отключ\w*|нет соединения|сбой обновления|ошибка лицензии)\b",
    re.IGNORECASE,
)
_BEHAVIORAL_CONTEXT_FIELDS = frozenset({
    "message", "request", "query", "query_name", "domain", "url", "uri", "path",
    "file", "process", "process_name", "process_path", "command", "image", "parent",
    "parent_process", "powershell", "registry", "registry_key", "task", "service",
    "user_agent", "dns", "cmdline", "command_line", "sha", "hash", "md5", "sha1",
    "sha256", "filename", "mail", "src_port", "dst_port", "port", "protocol",
    "proto", "record_type", "event_type", "action", "result", "object", "target",
    "network_direction",
})
_BEHAVIOR_RE = re.compile(
    r"\b(command|powershell|process|script|encoded|download|execute|spawn|inject|"
    r"credential|password|login|brute|phish|malware|ransom|tunnel|beacon|exfil|"
    r"registry|scheduled task|persistence|lateral|exploit|shell|dns quer|http request|"
    r"команд\w*|процесс\w*|скрипт\w*|загруз\w*|выполн\w*|парол\w*|туннел\w*)\b",
    re.IGNORECASE,
)


def build_natural_parts(incident: dict[str, Any]) -> list[str]:
    rule_name = incident.get("ruleName")
    title = incident.get("title")
    description = incident.get("description")
    parts: list[str] = []
    if isinstance(rule_name, str) and rule_name.strip():
        parts.append(rule_name.strip())
    if isinstance(title, str) and title.strip():
        title_clean = re.sub(r"^Correlation:\s*", "", title.strip(), flags=re.IGNORECASE)
        if not (isinstance(rule_name, str) and title_clean.casefold() == rule_name.strip().casefold()):
            parts.append(title_clean)
    if isinstance(description, str) and description.strip():
        parts.append(description.strip())
    tags = incident.get("tags")
    if isinstance(tags, list):
        tag_str = " ".join(str(t).strip() for t in tags if str(t).strip())
        if tag_str:
            parts.append(tag_str)
    if not parts:
        fallback = _fallback_semantic_label(incident)
        if fallback:
            parts.append(fallback)
    return _dedupe(parts)


def build_natural_text(incident: dict[str, Any]) -> str:
    return ". ".join(build_natural_parts(incident))


def build_technical_text(incident: dict[str, Any]) -> str:
    return extract_context_text(incident.get("context"))


def extract_context_text(context: Any) -> str:
    if not isinstance(context, dict):
        return ""
    seen_values: set[str] = set()
    values: list[str] = []
    for key, value in context.items():
        field = str(key).rsplit(".", 1)[-1].lower().strip()
        if field in _CONTEXT_DENYLIST or field not in _CONTEXT_ALLOWLIST:
            continue
        if not isinstance(value, (str, int, float, bool)):
            continue
        text = _sanitize(str(value))
        canonical = text.casefold()
        if not text or _is_noise_value(text) or canonical in seen_values:
            continue
        seen_values.add(canonical)
        values.append(f"{field}={text}")
    return " ".join(values)


def build_query_text(incident: dict[str, Any], translator: TranslatorProtocol | None = None) -> str:
    parts = build_natural_parts(incident)
    translated_parts = list(parts)
    if translator is not None:
        indices = [i for i, part in enumerate(parts) if _CYRILLIC_RE.search(part)]
        if indices:
            try:
                translated = translator.translate_batch([parts[i] for i in indices])
                for index, text in zip(indices, translated):
                    if text and text.strip():
                        translated_parts[index] = text.strip()
            except Exception:
                logger.exception("Translation failed; preserving original fields")
    technical = build_technical_text(incident)
    evidence = "; ".join(_dedupe(translated_parts))
    sections = [f"Observed security behavior: {evidence}" if evidence else ""]
    if technical:
        sections.append(f"Technical evidence: {technical}")
    return "\n".join(section for section in sections if section).strip()


def has_attack_behavior(incident: dict[str, Any], query: str) -> bool:
    natural = " ".join(build_natural_parts(incident))
    technical = build_technical_text(incident)
    combined = f"{natural} {technical} {query}".strip()
    if not combined:
        return False
    if _BEHAVIOR_RE.search(combined):
        return True
    technical_fields = {token.split("=", 1)[0] for token in technical.split() if "=" in token}
    has_behavioral_context = bool(technical_fields.intersection(_BEHAVIORAL_CONTEXT_FIELDS))
    if _OPERATIONAL_ONLY_RE.search(combined) and not has_behavioral_context:
        return False
    return has_behavioral_context


def _fallback_semantic_label(incident: dict[str, Any]) -> str:
    raw = " ".join(str(incident.get(key) or "") for key in ("ruleId", "sourceId"))
    tokens = [token for token in re.split(r"[^\w]+", raw.lower()) if token]
    meaningful = [token for token in tokens if token not in _GENERIC_ID_TOKENS and not token.isdigit()]
    return " ".join(dict.fromkeys(meaningful))


def _dedupe(parts: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        normalized = part.strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _sanitize(text: str) -> str:
    text = _IP_RE.sub(" ", text)
    text = _UUID_RE.sub(" ", text)
    text = _TS_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_noise_value(text: str) -> bool:
    return bool(_IP_RE.fullmatch(text) or _UUID_RE.fullmatch(text) or _TS_RE.fullmatch(text))
