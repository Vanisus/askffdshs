from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

# Keep this allowlist intentionally narrow. A URL merely ending in .crl is not
# sufficient evidence that the surrounding event is benign.
_PROVIDER_HOSTS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "Let's Encrypt",
        frozenset({"c.lencr.org"}),
    ),
    (
        "DigiCert",
        frozenset({
            "crl.digicert.com",
            "crl3.digicert.com",
            "crl4.digicert.com",
            "ocsp.digicert.com",
        }),
    ),
)


def detect_certificate_status_activity(incident: dict[str, Any]) -> dict[str, Any] | None:
    """Return an explainable benign-activity decision for known CA CRL/OCSP URLs.

    The detector is deliberately independent of ruleId and vendor field names.
    It scans string evidence recursively, but accepts only certificate-status
    artifacts hosted on a small allowlist of public CA infrastructure.
    """
    for text in _iter_strings(incident):
        for raw_url in _URL_RE.findall(text):
            url = raw_url.rstrip(".,;:!?)]}\\")
            try:
                parsed = urlsplit(url)
                host = (parsed.hostname or "").lower().rstrip(".")
            except ValueError:
                continue
            if not host:
                continue

            provider = _provider_for_host(host)
            artifact_type = _artifact_type(host, parsed.path)
            if provider is None or artifact_type is None:
                continue

            return {
                "source": "automatic",
                "abstained": True,
                "reason": "certificate_revocation_check",
                "explanation": (
                    f"Observed a {artifact_type} request to {provider} certificate-status "
                    f"infrastructure ({host}). This event alone contains no observable "
                    "adversary behavior, so ATT&CK mapping was not applied."
                ),
                "context": {
                    "activity_type": "certificate_revocation_check",
                    "provider": provider,
                    "endpoint": host,
                    "artifact_type": artifact_type,
                },
            }
    return None


def _provider_for_host(host: str) -> str | None:
    # Let's Encrypt publishes CRLs on numbered subdomains such as r11.c.lencr.org.
    if host == "c.lencr.org" or host.endswith(".c.lencr.org"):
        return "Let's Encrypt"
    for provider, hosts in _PROVIDER_HOSTS:
        if host in hosts:
            return provider
    return None


def _artifact_type(host: str, path: str) -> str | None:
    path_lower = path.lower()
    first_label = host.split(".", 1)[0]
    if path_lower.endswith(".crl") or first_label.startswith("crl") or host.endswith(".c.lencr.org"):
        return "CRL"
    if first_label == "ocsp" or "/ocsp" in path_lower:
        return "OCSP"
    return None


def _iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_strings(nested)
