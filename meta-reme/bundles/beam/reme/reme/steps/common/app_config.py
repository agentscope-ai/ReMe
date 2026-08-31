"""Return the effective application configuration."""

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..base_step import BaseStep
from ...components import R

_SECRET_KEYS = {
    "api_key",
    "app_secret",
    "authorization",
    "cookie",
    "password",
    "private_key",
    "proxy_authorization",
    "secret",
    "set_cookie",
    "token",
}
_SECRET_SUFFIXES = ("_access_key", "_api_key", "_password", "_private_key", "_secret", "_token")


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES)


def _redact_url(value: str) -> str:
    """Redact credentials embedded in an HTTP-style URL."""
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.netloc:
            return value

        netloc = parsed.netloc
        if parsed.password is not None:
            host = parsed.hostname or ""
            if ":" in host:
                host = f"[{host}]"
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            netloc = f"{parsed.username or ''}:***@{host}"

        query = urlencode(
            [
                (key, "***" if _is_secret_key(key) else item)
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            ],
        )
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    except ValueError:
        return value


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_string = str(key)
            if _is_secret_key(key_string) or (
                key_string.lower() == "credential" and not isinstance(item, (dict, list))
            ):
                redacted[key] = "***"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _redact_url(value)
    return value


@R.register("app_config_step")
class AppConfigStep(BaseStep):
    """Return the effective config without startup environment or secrets."""

    async def execute(self):
        assert self.context is not None
        assert self.app_context is not None

        config = self.app_context.app_config.model_dump(mode="json", exclude={"environment"})
        self.context.response.answer = _redact(config)
        return self.context.response
