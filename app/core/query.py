import logging
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TranslatorProtocol(Protocol):
    def translate(self, text: str) -> str: ...
    def translate_batch(self, texts: list[str]) -> list[str]: ...


_CONTEXT_ALLOWLIST = (
    "message", "request", "query", "query_name", "hostname", "mail_host",
    "host", "domain", "url", "uri", "path", "file", "process", "command",
    "image", "parent", "powershell", "registry", "task", "service",
    "user_agent", "dns", "cmdline", "command_line", "subject",
    "account", "sha", "hash", "md5", "sha256", "filename", "mail",
    "src", "dst", "port", "protocol", "proto", "user", "user_name",
)

_CONTEXT_DENYLIST = (
    "event_id", "alert_id", "rule_id", "timestamp",
    "asset.", "criticality", "owner", "department",
    "conn_count", "count", "@timestamp", "severity",
)


_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){1,3}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:?\d{2})?\b",
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
        title_clean = title.strip()
        if title_clean.startswith("Correlation:"):
            title_clean = title_clean[len("Correlation:"):].strip()

        if not (
            isinstance(rule_name, str)
            and title_clean.lower() == rule_name.strip().lower()
        ):
            parts.append(title_clean)

    if isinstance(description, str) and description.strip():
        parts.append(description.strip())

    tags = incident.get("tags")
    if isinstance(tags, list):
        tag_str = " ".join(str(t).strip() for t in tags if str(t).strip())
        if tag_str:
            parts.append(tag_str)

    return parts


def build_natural_text(incident: dict[str, Any]) -> str:
    return ". ".join(build_natural_parts(incident))

def build_technical_text(incident: dict[str, Any]) -> str:
    parts: list[str] = []

    rule_id = incident.get("ruleId")
    if isinstance(rule_id, str) and rule_id.strip():
        parts.append(rule_id.strip())

    ctx = extract_context_text(incident.get("context"))
    if ctx:
        parts.append(ctx)

    return " ".join(parts)


def extract_context_text(context: Any) -> str:
    if not isinstance(context, dict):
        return ""

    seen: set[str] = set()
    values: list[str] = []

    for key, value in context.items():
        key_l = str(key).lower()

        if any(blocked in key_l for blocked in _CONTEXT_DENYLIST):
            continue
        if not any(allowed in key_l for allowed in _CONTEXT_ALLOWLIST):
            continue
        if not isinstance(value, (str, int, float)):
            continue

        text = _sanitize(str(value))
        if not text or _is_noise_value(text):
            continue
        if text in seen:
            continue

        seen.add(text)
        values.append(text)

    return " ".join(values)

def build_query_text(
    incident: dict[str, Any],
    translator: TranslatorProtocol | None = None,
) -> str:
    parts = build_natural_parts(incident)
    technical = build_technical_text(incident)

    translated_parts: list[str] = []
    if translator is not None and parts:
        try:
            raw = translator.translate_batch(parts)
            translated_parts = [t for t in raw if t and t.strip()]
        except Exception:
            logger.exception("Translation failed for parts: %r", parts)
            translated_parts = []

    if not translated_parts:
        translated_parts = parts

    final = " ".join([*translated_parts, technical])
    return final.strip()

def _sanitize(text: str) -> str:
    text = _IP_RE.sub(" ", text)
    text = _UUID_RE.sub(" ", text)
    text = _TS_RE.sub(" ", text)
    return text.strip()


def _is_noise_value(text: str) -> bool:
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){2,3}", text):
        return True
    if _UUID_RE.fullmatch(text):
        return True
    if _TS_RE.fullmatch(text):
        return True
    return False
