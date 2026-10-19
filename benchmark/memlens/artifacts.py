"""Small, credential-free snapshots and diagnostics for benchmark artifacts."""

import os
from pathlib import Path
import re
import traceback
from urllib.parse import urlsplit, urlunsplit

_PRIVATE_KEY = re.compile(
    r"credential|api[_-]?key|private[_-]?key|authorization|password|secret|cookie|access[_-]?token|refresh[_-]?token|"
    r"(^|[_-])token$|headers|request[_-]?body|^environment$|^config(?:uration)?$|^messages$",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|private[_-]?key|authorization|password|secret|(?:access|refresh|id)[_-]?token|token)"
    r"[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
)
_MODEL_FIELDS = (
    "backend",
    "model",
    "stream",
    "context_size",
    "max_retries",
    "retry_delay",
)
_MODEL_OPTIONS = {
    "parameters": (
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "thinking_enable",
    ),
    "extra_body": ("enable_thinking",),
    "client_kwargs": ("timeout", "max_retries"),
}


def _safe_text(value: str) -> str:
    for key, secret in sorted(os.environ.items(), key=lambda item: len(item[1]), reverse=True):
        if secret and _PRIVATE_KEY.search(key):
            value = value.replace(secret, "[REDACTED]")

    def safe_url(match):
        try:
            url = urlsplit(match.group())
            host = url.netloc.rsplit("@", 1)[-1]
            return urlunsplit((url.scheme, host, url.path, "[REDACTED]" if url.query else "", ""))
        except ValueError:
            return "[REDACTED_URL]"

    value = re.sub(r"https?://[^\s\"'<>]+", safe_url, value)
    value = re.sub(r"(?s)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[REDACTED_KEY]", value)
    value = re.sub(r"(?im)(\b(?:set[-_])?cookie[\"']?\s*[:=]\s*)[^\r\n]+", r"\1[REDACTED]", value)
    value = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", "[REDACTED_AUTH]", value)
    value = _ASSIGNMENT.sub(r"\1[REDACTED]", value)
    value = re.sub(r"(?i)data:[^;,\s]+;base64,[A-Za-z0-9+/=\s]+", "[REDACTED_IMAGE]", value)
    value = re.sub(r"\b[A-Za-z0-9+/]{80,}={0,2}", "[REDACTED_DATA]", value)
    # Exceptions sometimes embed complete request/config dictionaries in their text.
    return re.sub(
        r"(?is)(\b(?:config|configuration|request_headers|headers|request_body|messages)\b[\"']?\s*[:=]).*",
        r"\1[REDACTED]",
        value,
    )


def safe_value(value):
    """Redact diagnostic data recursively without serializing arbitrary objects."""
    if isinstance(value, dict):
        return {
            _safe_text(str(key)): (
                "[REDACTED]"
                if _PRIVATE_KEY.search(str(key)) or (key == "data" and value.get("type") == "base64")
                else safe_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [safe_value(item) for item in value]
    if isinstance(value, str):
        return _safe_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


def error_details(exc: BaseException, seen: set[int] | None = None) -> dict:
    """Keep the error chain and frame locations, never frame locals or source text."""
    seen = set() if seen is None else seen
    seen.add(id(exc))
    result = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": [
            {
                "file": Path(frame.filename).name,
                "line": frame.lineno,
                "function": frame.name,
            }
            for frame in traceback.extract_tb(exc.__traceback__)
        ],
    }
    cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    if cause is not None and id(cause) not in seen:
        result["cause"] = error_details(cause, seen)
    return safe_value(result)


def model_settings(config) -> dict:
    """Whitelist reproducibility fields from a validated ApplicationConfig."""
    models = {}
    for name, component in config.components.get("as_llm", {}).items():
        spec = component.model_dump()
        snapshot = {key: spec[key] for key in _MODEL_FIELDS if key in spec}
        for key, allowed in _MODEL_OPTIONS.items():
            options = spec.get(key) or {}
            snapshot[key] = {name: options[name] for name in allowed if name in options}
        models[name] = snapshot
    return safe_value(models)


def judge_settings(app) -> dict:
    """Record the actual plugin/env-expanded judge model and official protocol."""
    from judge_memlens.protocol import PROTOCOL_ID, SOURCE_REVISION

    return {
        "protocol_id": PROTOCOL_ID,
        "source_revision": SOURCE_REVISION,
        "models": model_settings(app.config),
    }
