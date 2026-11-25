"""Complete user settings imported once and applied before each worker starts."""

import json
import ipaddress
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from ..components.asr.dashscope_asr import DEFAULT_DASHSCOPE_ASR_MODEL
from ..components.service.cli_service import prepare_start_config
from ..components.tts.dashscope_tts import DEFAULT_DASHSCOPE_TTS_MODEL, DEFAULT_DASHSCOPE_TTS_VOICE
from ..schema import ApplicationConfig
from ..utils.dashscope_speech import DEFAULT_DASHSCOPE_BASE_URL
from ..utils.env_utils import load_env

SECRET_FIELDS = ("api_key", "dashscope_api_key")
_MODEL_OVERRIDES = ("model", "review_model", "memories.extract_model", "memories.consolidation_model")


class Settings(BaseModel):
    """The complete values owned by a managed ReMe instance."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    workspace_dir: str | None = Field(default=None, max_length=4096)
    timezone: str | None = Field(default=None, max_length=128)
    language: str = Field(default="zh", max_length=32)
    harness: Literal["codex"] | None = None
    base_url: str | None = Field(default=None, max_length=2048)
    model: str | None = Field(default=None, max_length=256)
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    api_key: SecretStr | None = Field(default=None, max_length=4096)
    dashscope_api_key: SecretStr | None = Field(default=None, max_length=4096)
    asr_enabled: bool = False
    tts_enabled: bool = False
    speech_base_url: str = Field(default=DEFAULT_DASHSCOPE_BASE_URL, max_length=2048)
    asr_model: str = Field(default=DEFAULT_DASHSCOPE_ASR_MODEL, max_length=256)
    tts_model: str = Field(default=DEFAULT_DASHSCOPE_TTS_MODEL, max_length=256)
    tts_voice: str = Field(default=DEFAULT_DASHSCOPE_TTS_VOICE, max_length=256)

    @field_validator("workspace_dir")
    @classmethod
    def workspace_path(cls, value):
        """Persist an absolute backend path so a later restart resolves the same directory."""
        return str(Path(value).expanduser().resolve()) if value else None

    @field_validator("base_url", "speech_base_url")
    @classmethod
    def provider_url(cls, value):
        """Validate an address to which provider routes can be appended."""
        if not value:
            return value
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.port == 0:
            raise ValueError("Expected an HTTP or HTTPS base URL with a valid host")
        if (
            url.username is not None
            or url.password is not None
            or any(character in value for character in "?#")
            or any(character.isspace() for character in value)
        ):
            raise ValueError("Expected an HTTP or HTTPS base URL without credentials, query or fragment")
        if url.scheme == "http" and url.hostname != "localhost":
            try:
                local = ipaddress.ip_address(url.hostname).is_loopback
            except ValueError:
                local = False
            if not local:
                raise ValueError("Remote model and speech APIs require HTTPS")
        return value.rstrip("/")

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value):
        """Validate the schedule timezone before restarting any resources."""
        if value:
            try:
                ZoneInfo(value)
            except (ValueError, ZoneInfoNotFoundError) as error:
                raise ValueError("Expected an IANA timezone") from error
        return value or None

    @field_validator("api_key", "dashscope_api_key", mode="before")
    @classmethod
    def strip_key(cls, value):
        """Remove whitespace copied alongside a credential."""
        return value.strip() if isinstance(value, str) else value

    def private_values(self) -> dict:
        """Serialize plaintext only for encryption and the private child pipe."""
        values = self.model_dump(mode="json", exclude=set(SECRET_FIELDS))
        for field in SECRET_FIELDS:
            secret = getattr(self, field)
            values[field] = secret.get_secret_value() if secret else None
        return values

    def public_values(self) -> dict:
        """Expose credential presence while retaining all editable ordinary values."""
        values = self.model_dump(mode="json", exclude=set(SECRET_FIELDS))
        values.update({f"{field}_set": bool(getattr(self, field)) for field in SECRET_FIELDS})
        return values

    def missing_fields(self) -> list[str]:
        """Return the effective values required by the enabled features."""
        required = ["workspace_dir"]
        if self.harness == "codex":
            required.extend(("base_url", "model", "api_key"))
        if self.asr_enabled or self.tts_enabled:
            required.extend(("speech_base_url", "dashscope_api_key"))
        if self.asr_enabled:
            required.append("asr_model")
        if self.tts_enabled:
            required.extend(("tts_model", "tts_voice"))
        return [field for field in required if not getattr(self, field)]


class SettingsRequest(Settings):
    """A revisioned patch; omitted credentials retain their current values."""

    revision: int = Field(ge=0)


class Profile(BaseModel):
    """One successfully started generation of the complete user settings."""

    revision: int = Field(default=0, ge=0)
    settings: Settings = Field(default_factory=Settings)


def _public_url(value: str | None) -> str | None:
    if not value:
        return value
    url = urlsplit(value)
    return urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))


def _initial_settings(config: dict) -> Settings:
    typed = ApplicationConfig.model_validate(config)
    components = config.get("components", {})
    wrapper = components.get("agent_wrapper", {}).get("default", {})
    server = components.get("codex_app_server", {}).get(wrapper.get("app_server", "default"), {})
    speech = {kind: components.get(kind, {}).get("default", {}) for kind in ("asr", "tts")}
    asr, tts = speech.values()
    editable = wrapper.get("backend") == "codex" and server.get("auth_mode") != "oauth"
    provider = server
    if wrapper.get("backend") == "agentscope":
        llm = components.get("as_llm", {}).get(wrapper.get("as_llm", "default"), {})
        provider = llm.get("credential") or {}
    elif wrapper.get("backend") == "claude_code":
        relay = components.get("anthropic_relay", {}).get(wrapper.get("anthropic_relay", "default"), {})
        provider = {"api_key": wrapper.get("api_key"), "base_url": relay.get("base_url", wrapper.get("base_url"))}
    key = next((item.get("api_key") for item in speech.values() if item.get("api_key")), None)
    if not key:
        key = next(
            (
                step.get("api_key")
                for job in config.get("jobs", {}).values()
                for step in job.get("steps", [])
                if step.get("backend") == "dashscope_search_step" and step.get("api_key")
            ),
            None,
        )
    key = key or os.getenv("DASHSCOPE_API_KEY") or None
    jobs = typed.jobs
    return Settings(
        workspace_dir=typed.workspace_dir,
        timezone=typed.timezone,
        language=typed.language,
        harness="codex" if editable else None,
        base_url=_public_url(provider.get("base_url")),
        model=llm.get("model") if wrapper.get("backend") == "agentscope" else wrapper.get("model"),
        effort=wrapper.get("effort") if editable else None,
        api_key=provider.get("api_key"),
        dashscope_api_key=key,
        asr_enabled=bool(key and asr.get("backend") == "dashscope" and "asr" in jobs and jobs["asr"].enabled),
        tts_enabled=bool(key and tts.get("backend") == "dashscope" and "tts" in jobs and jobs["tts"].enabled),
        speech_base_url=_public_url(asr.get("base_url") or tts.get("base_url")) or DEFAULT_DASHSCOPE_BASE_URL,
        asr_model=asr.get("model", DEFAULT_DASHSCOPE_ASR_MODEL),
        tts_model=tts.get("model", DEFAULT_DASHSCOPE_TTS_MODEL),
        tts_voice=(tts.get("parameters") or {}).get("voice", tts.get("voice", DEFAULT_DASHSCOPE_TTS_VOICE)),
    )


def resolve_worker_config(arguments: dict, settings: Settings | None):
    """Import startup inputs once, then use the complete saved managed values."""
    # Deployment-only variables still feed YAML expansion. Managed values below
    # are complete snapshots and always replace their initialization inputs.
    environment = load_env(override=False)
    for secret in ("REME_ENCRYPTION_KEY", "REME_ADMIN_TOKEN"):
        environment.pop(secret, None)
        os.environ.pop(secret, None)
    if settings is not None:
        variables = {
            "DASHSCOPE_API_KEY": settings.dashscope_api_key.get_secret_value() if settings.dashscope_api_key else "",
        }
        if settings.harness == "codex":
            variables.update(
                LLM_API_KEY=settings.api_key.get_secret_value() if settings.api_key else "",
                LLM_BASE_URL=settings.base_url or "",
                REME_MODEL_NAME=settings.model or "",
            )
        for key, value in variables.items():
            os.environ[key] = value
        environment.update(variables)
    config = prepare_start_config(arguments)
    settings = settings or _initial_settings(config)
    if settings.workspace_dir:
        workspace = Path(settings.workspace_dir).expanduser().resolve()
        parent = next((path for path in (workspace, *workspace.parents) if path.exists()), None)
        if parent is None or not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            raise ValueError("Workspace directory is not writable")
        settings.workspace_dir = str(workspace)
    config.update(workspace_dir=settings.workspace_dir, timezone=settings.timezone, language=settings.language)
    config["environment"] = {**config.get("environment", {}), **environment}
    for secret in ("REME_ENCRYPTION_KEY", "REME_ADMIN_TOKEN"):
        config["environment"].pop(secret, None)
    components = config.get("components", {})
    wrapper = components.get("agent_wrapper", {}).get("default", {})
    server = components.get("codex_app_server", {}).get(wrapper.get("app_server", "default"), {})
    editable = (
        wrapper.get("backend") == "codex" and server.get("backend") == "sdk" and server.get("auth_mode") != "oauth"
    )
    if settings.harness and not editable:
        raise ValueError("This ReMe configuration does not use the selected Harness")
    if editable:
        server.update(
            base_url=settings.base_url or "",
            api_key=settings.api_key.get_secret_value() if settings.api_key else "",
        )
        wrapper.update(model=settings.model, effort=settings.effort)
        options = dict.fromkeys(_MODEL_OVERRIDES, settings.model)
        options["cli_auth_credentials_store"] = "ephemeral"
        if settings.effort is not None:
            options["model_reasoning_effort"] = settings.effort
        retained = [
            item
            for item in server.get("config_overrides", [])
            if item.partition("=")[0].strip() not in {*options, "model_reasoning_effort"}
        ]
        server["config_overrides"] = retained + [
            f"{key}={json.dumps(value)}" for key, value in options.items() if value is not None
        ]
        catalog = server.get("model_catalog")
        if catalog and all(item.get("slug") != settings.model for item in catalog.get("models", [])):
            server.pop("model_catalog")
    elif wrapper.get("backend") == "claude_code":
        wrapper.update(model=settings.model, api_key=settings.api_key.get_secret_value() if settings.api_key else "")
        relay = components.get("anthropic_relay", {}).get(wrapper.get("anthropic_relay", "default"), {})
        if relay:
            relay["base_url"] = settings.base_url or ""
        else:
            wrapper["base_url"] = settings.base_url or ""
    elif wrapper.get("backend") == "agentscope":
        llm = components.get("as_llm", {}).get(wrapper.get("as_llm", "default"), {})
        if llm:
            llm["model"] = settings.model
            llm["credential"] = {
                **(llm.get("credential") or {}),
                "api_key": settings.api_key.get_secret_value() if settings.api_key else "",
                "base_url": settings.base_url or "",
            }
    speech_available = {}
    for kind in ("asr", "tts"):
        component = components.get(kind, {}).get("default", {})
        speech_available[kind] = component.get("backend") == "dashscope" and kind in config.get("jobs", {})
        enabled = getattr(settings, f"{kind}_enabled")
        if enabled and not speech_available[kind]:
            raise ValueError(f"This ReMe configuration does not include DashScope {kind.upper()}")
        if component.get("backend") == "dashscope":
            component.update(
                api_key=settings.dashscope_api_key.get_secret_value() if settings.dashscope_api_key else "",
                base_url=settings.speech_base_url,
                model=getattr(settings, f"{kind}_model"),
            )
            if kind == "tts":
                component["voice"] = settings.tts_voice
                component["parameters"] = {
                    key: value for key, value in (component.get("parameters") or {}).items() if key != "voice"
                }
        if kind in config.get("jobs", {}):
            config["jobs"][kind]["enabled"] = enabled
    search = []
    for job in config.get("jobs", {}).values():
        for section in ("steps", "init_steps"):
            search.extend(step for step in job.get(section, []) if step.get("backend") == "dashscope_search_step")
    for step in search:
        step["api_key"] = settings.dashscope_api_key.get_secret_value() if settings.dashscope_api_key else ""
    view = {
        **settings.public_values(),
        "model_editable": editable,
        "available_harnesses": ["codex"] if editable else [],
        "dashscope_editable": any(speech_available.values()) or bool(search),
        "asr_available": speech_available["asr"],
        "tts_available": speech_available["tts"],
        "missing_fields": settings.missing_fields(),
        "defaults": {
            "speech_base_url": DEFAULT_DASHSCOPE_BASE_URL,
            "asr_model": DEFAULT_DASHSCOPE_ASR_MODEL,
            "tts_model": DEFAULT_DASHSCOPE_TTS_MODEL,
            "tts_voice": DEFAULT_DASHSCOPE_TTS_VOICE,
        },
    }
    return config, settings, view
