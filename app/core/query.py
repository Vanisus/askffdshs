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
    "command", "image", "parent", "powershell", "registry", "task", "service",
    "user_agent", "dns", "cmdline", "command_line", "subject", "account",
    "sha", "hash", "md5", "sha256", "filename", "mail", "src", "dst",
    "port", "protocol", "proto", "user", "user_name", "total_events",
    "affected_hosts", "record_type", "event_type", "action", "result",
})
_CONTEXT_DENYLIST = frozenset({
    "event_id", "alert_id", "rule_id", "timestamp", "@timestamp",
    "criticality", "owner", "department", "conn_count", "count", "severity",
})
_IP_RE = re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){1,3}(?![\w.])")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:?\d{2})?\b", re.IGNORECASE)
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_GENERIC_ID_TOKENS = frozenset({"rule", "alert", "correlation", "event"})


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
    return parts


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
        if not text or _is_noise_value(text) or text in seen_values:
            continue
        seen_values.add(text)
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
    return " ".join([*translated_parts, technical]).strip()


def _fallback_semantic_label(incident: dict[str, Any]) -> str:
    """Humanize identifiers only for schema-poor input; never map IDs to ATT&CK."""
    raw = " ".join(str(incident.get(key) or "") for key in ("ruleId", "sourceId"))
    tokens = [token for token in re.split(r"[^\w]+", raw.lower()) if token]
    meaningful = [token for token in tokens if token not in _GENERIC_ID_TOKENS and not token.isdigit()]
    return " ".join(dict.fromkeys(meaningful))


def _sanitize(text: str) -> str:
    text = _IP_RE.sub(" ", text)
    text = _UUID_RE.sub(" ", text)
    text = _TS_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_noise_value(text: str) -> bool:
    return bool(_IP_RE.fullmatch(text) or _UUID_RE.fullmatch(text) or _TS_RE.fullmatch(text))
