"""Return the effective application configuration."""

from typing import Any
from hashlib import sha256
import json
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ...constants import DEFAULT_MEMORY_TAG_KEY
from ..base_step import BaseStep
from ...core.registry import R
from ...schema.agent_view import ClientInfo

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


@R.register("client_info_step")
class ClientInfoStep(BaseStep):
    """Publish client capabilities without exposing component configuration."""

    async def execute(self):
        config = self.app_context.app_config
        allowlist = getattr(self.app_context.service, "jobs", None)
        jobs = sorted(
            name
            for name, job in config.jobs.items()
            if job.enabled
            and job.enable_serve
            and job.backend != "background"
            and (allowlist is None or name in allowlist)
        )
        components = config.components
        store = components.get("file_store", {}).get("default")
        index_name = getattr(store, "tag_index", "default")
        index = components.get("tag_index", {}).get(index_name)
        tts = components.get("tts", {}).get("default")
        speech = (
            {
                key: value
                for key, value in tts.model_dump().items()
                if key
                in {
                    "backend",
                    "base_url",
                    "model",
                    "voice",
                    "audio_format",
                    "sample_rate",
                    "parameters",
                }
            }
            if tts
            else {}
        )
        self.context.response.answer = ClientInfo(
            workspace_dir=config.workspace_dir,
            timezone=config.timezone,
            resource_dir=config.resource_dir,
            daily_dir=config.daily_dir,
            digest_dir=config.digest_dir,
            tag_key=getattr(index, "tag_key", DEFAULT_MEMORY_TAG_KEY),
            jobs=jobs,
            asr="asr" in jobs,
            tts="tts" in jobs,
            speech_revision=sha256(json.dumps(speech, sort_keys=True).encode()).hexdigest(),
        ).model_dump(mode="json")
        return self.context.response
