"""Small, credential-safe diagnostics and accounting helpers for this runner."""

import os
from pathlib import Path, PurePosixPath
import re
import traceback

USAGE_POLICY = {
    "version": 2,
    "scope": "observed successful HTTP requests, including failed candidate flows and compression; judge excluded",
    "memory_sources": ["users/*/sessions/*.json", "users/*/index-response.json"],
    "answer_source": "users/*/questions/*/answer-response.json",
    "means": "observed run cost amortized over scheduled users / expected questions, not successful samples",
}
_SENSITIVE = re.compile(
    r"authorization|credential|secret|password|api.?key|private[_-]?key|cookie|headers|"
    r"(?:^|[_-])(?:access[_-]?token|refresh[_-]?token|id[_-]?token|token)$",
    re.I,
)
_USAGE_FIELDS = {
    "usage",
    "coverage",
    "records",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "noncached_input_tokens",
    "reasoning_tokens",
}


def safe_text(value, limit: int = 1200) -> str:
    """Retain short diagnostic text, without credentials, URLs or binary payloads."""
    text = str(value)
    for key, secret in os.environ.items():
        if secret and (_SENSITIVE.search(key) or re.search(r"(?:^|_)TOKEN(?:$|_)", key.upper())):
            if len(secret) < 12:
                text = re.sub(r"(?<![\w-])" + re.escape(secret) + r"(?![\w-])", "[redacted]", text)
            else:
                text = text.replace(secret, "[redacted]")
    text = re.sub(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
        "[private key redacted]",
        text,
        flags=re.S,
    )
    text = re.sub(r"(?i)\b(?:request[_ -]?)?headers\s*[:=].*", "headers=[redacted]", text)
    text = re.sub(r"(?i)\b(?:set-cookie|cookie)\s*:[^\r\n]*", "cookie: [redacted]", text)
    text = re.sub(r"data:[^\s,;]+(?:;[^\s,]*)?,[^\s\"']+", "[image data redacted]", text, flags=re.I)
    text = re.sub(r"https?://[^\s\"'<>]+", "[url redacted]", text, flags=re.I)
    text = re.sub(r"(?<![\w])(?:[A-Za-z]:\\|/)[^\s\"']+", "[path redacted]", text)
    text = re.sub(r"\b(?:Bearer|Basic)\s+[^\s,;\"'}]+", "[authorization redacted]", text, flags=re.I)
    text = re.sub(
        r"(?i)([\"']?(?:api[_-]?key|private[_-]?key|authorization|password|secret|access[_-]?token|"
        r"refresh[_-]?token|id[_-]?token|token|cookie)[\"']?\s*[:=]\s*)"
        r"(?:[\"'][^\"']*[\"']|[^\s,;}]+)",
        r"\1[redacted]",
        text,
    )
    text = re.sub(r"\b(?:sk-|sk_|ghp_|github_pat_)[A-Za-z0-9_-]+", "[key redacted]", text)
    text = re.sub(r"[A-Za-z0-9+/=_-]{80,}", "[payload redacted]", text)
    return text[:limit]


def safe_value(value, depth: int = 0, *, fixed_keys=()):
    """Sanitize failed response envelopes, never serialize arbitrary objects/locals."""
    if depth > 6:
        return "[nested data omitted]"
    if isinstance(value, dict):
        return {
            (key if key in fixed_keys else safe_text(key, 100)): (
                "[redacted]"
                if _SENSITIVE.search(str(key)) or str(key).lower() in {"base64", "data", "body", "request", "traceback"}
                else safe_value(item, depth + 1, fixed_keys=fixed_keys)
            )
            for key, item in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple)):
        return [safe_value(item, depth + 1, fixed_keys=fixed_keys) for item in value[:20]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return safe_text(value) if isinstance(value, str) else f"[{type(value).__name__} omitted]"


def response_record(response):
    """Keep successful response artifacts compatible; sanitize failed envelopes."""
    if response is None:
        return None
    value = response.model_dump()
    if response.success:
        return value
    return {
        **safe_value({key: item for key, item in value.items() if key not in {"success", "answer", "metadata"}}),
        "success": value["success"],
        "answer": safe_value(value["answer"]),
        "metadata": safe_value(value["metadata"], fixed_keys=_USAGE_FIELDS),
    }


def safe_image_path(source) -> str:
    """Only retain bounded dataset-relative image identifiers in diagnostics."""
    source = str(source)
    path = PurePosixPath(source)
    if not source or len(source) > 300 or path.is_absolute() or ".." in path.parts:
        return "[invalid image path]"
    if any(character in source for character in ("\\", ":", "\n", "\r")):
        return "[invalid image path]"
    return safe_text(source, 300)


class ImageInputError(ValueError):
    """Attach only the dataset-relative image identifier to a preprocessing error."""

    def __init__(self, source, error):
        super().__init__(safe_text(error))
        self.image_path = safe_image_path(source)


def error_record(error, *, stage: str, user_id=None, session_id=None, question_id=None, response=None) -> dict:
    """Preserve stage, a bounded cause chain and stack locations, never frame locals."""
    chain, seen = [], set()
    current = error
    image_path = None
    while current is not None and id(current) not in seen and len(chain) < 5:
        seen.add(id(current))
        image_path = image_path or getattr(current, "image_path", None)
        chain.append(
            {
                "type": type(current).__name__,
                "reason": safe_text(current),
                "traceback": [
                    {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                    for frame in traceback.extract_tb(current.__traceback__)[-12:]
                ],
            },
        )
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    reason = response.answer if response is not None and not response.success else str(error)
    return {
        "stage": stage,
        "user_id": safe_text(user_id, 120) if user_id is not None else None,
        "session_id": safe_text(session_id, 120) if session_id is not None else None,
        "question_id": safe_text(question_id, 120) if question_id is not None else None,
        "image_path": image_path,
        "error_type": type(error).__name__,
        "reason": safe_value(reason),
        "chain": chain,
    }


def mean_tokens(total: dict, denominator: int) -> dict:
    """Amortize observed counters over the explicit scheduled population."""
    return {
        key: total[key] / denominator if denominator and total[key] is not None else None
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "noncached_input_tokens")
    }


def judge_settings(app) -> dict:
    """Export a whitelist from the validated application, not unresolved YAML/env text."""
    from judge_mobilemem.protocol import MEMBASE_REVISION, SOURCE_REVISION, protocol_id

    model = app.config.components["as_llm"]["judge"].model_dump()
    mode = app.config.jobs["answer_judge"].model_dump().get("judge_protocol", "membase_default")
    fields = ("backend", "model", "stream", "context_size", "max_retries", "retry_delay")
    return {
        **{key: safe_value(model[key]) for key in fields if key in model},
        "parameters": {
            key: safe_value(value)
            for key, value in (model.get("parameters") or {}).items()
            if key in {"temperature", "max_tokens", "thinking_enable", "top_p"}
        },
        "extra_body": {"enable_thinking": (model.get("extra_body") or {}).get("enable_thinking")},
        "client_kwargs": {
            key: value for key, value in (model.get("client_kwargs") or {}).items() if key in {"timeout", "max_retries"}
        },
        "protocol": protocol_id(mode),
        "protocol_mode": mode,
        "mobilemem_revision": SOURCE_REVISION,
        "membase_revision": MEMBASE_REVISION,
    }
