"""LLM model wrappers for AgentScope."""

from typing import Literal

from agentscope.credential import (
    AnthropicCredential,
    CredentialBase,
    DashScopeCredential,
    DeepSeekCredential,
    GeminiCredential,
    MoonshotCredential,
    OllamaCredential,
    OpenAICredential,
    XAICredential,
)
from agentscope.model import ChatModelBase
from pydantic import ConfigDict, Field, field_validator

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsLLM(BaseComponent):
    """Base wrapper for AgentScope chat models.

    Subclasses set ``credential_cls`` and inherit ``_start`` / ``_close``.
    """

    component_type = ComponentEnum.AS_LLM
    credential_cls: type[CredentialBase]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: ChatModelBase | None = None

    async def _start(self) -> None:
        if self.model is not None:
            return
        kwargs = dict(self.kwargs)
        credential = self.credential_cls(**kwargs.pop("credential", {}))
        model_cls = credential.get_chat_model_class()
        params_dict = kwargs.pop("parameters", None)
        parameters = model_cls.Parameters(**params_dict) if params_dict else None
        self.model = model_cls(credential=credential, parameters=parameters, **kwargs)


@R.register("openai")
class OpenAIAsLLM(BaseAsLLM):
    """OpenAI chat model wrapper."""

    credential_cls = OpenAICredential


_ORCAROUTER_BASE_URL = "https://api.orcarouter.ai/v1"


class OrcaRouterCredential(OpenAICredential):
    """OpenAI-compatible credential for the OrcaRouter gateway.

    OrcaRouter exposes an OpenAI-compatible ``chat/completions`` endpoint at
    ``https://api.orcarouter.ai/v1``. The shared LLM config passes
    ``base_url: ${LLM_BASE_URL:-}``, so an unset ``LLM_BASE_URL`` yields an
    empty string; this credential falls back to the hosted gateway in that
    case while still allowing a custom/self-hosted endpoint.
    """

    model_config = ConfigDict(
        title="OrcaRouter API",
    )

    type: Literal["orcarouter_credential"] = "orcarouter_credential"
    """The credential type."""

    base_url: str = Field(
        default=_ORCAROUTER_BASE_URL,
        description="The base URL for the OrcaRouter API.",
    )
    """The base URL for the OrcaRouter API."""

    @field_validator("base_url", mode="before")
    @classmethod
    def _default_base_url(cls, value: str | None) -> str:
        """Fall back to the hosted gateway when the shared config leaves the field empty."""
        return value or _ORCAROUTER_BASE_URL


@R.register("orcarouter")
class OrcaRouterAsLLM(BaseAsLLM):
    """OrcaRouter chat model wrapper (OpenAI-compatible gateway)."""

    credential_cls = OrcaRouterCredential


@R.register("anthropic")
class AnthropicAsLLM(BaseAsLLM):
    """Anthropic chat model wrapper."""

    credential_cls = AnthropicCredential


@R.register("dashscope")
class DashScopeAsLLM(BaseAsLLM):
    """DashScope chat model wrapper."""

    credential_cls = DashScopeCredential


@R.register("deepseek")
class DeepSeekAsLLM(BaseAsLLM):
    """DeepSeek chat model wrapper."""

    credential_cls = DeepSeekCredential


@R.register("gemini")
class GeminiAsLLM(BaseAsLLM):
    """Gemini chat model wrapper."""

    credential_cls = GeminiCredential


@R.register("moonshot")
class MoonshotAsLLM(BaseAsLLM):
    """Moonshot chat model wrapper."""

    credential_cls = MoonshotCredential


@R.register("ollama")
class OllamaAsLLM(BaseAsLLM):
    """Ollama chat model wrapper."""

    credential_cls = OllamaCredential


@R.register("xai")
class XAIAsLLM(BaseAsLLM):
    """xAI chat model wrapper."""

    credential_cls = XAICredential


__all__ = [
    "BaseAsLLM",
    "OpenAIAsLLM",
    "OrcaRouterAsLLM",
    "AnthropicAsLLM",
    "DashScopeAsLLM",
    "DeepSeekAsLLM",
    "GeminiAsLLM",
    "MoonshotAsLLM",
    "OllamaAsLLM",
    "XAIAsLLM",
]
