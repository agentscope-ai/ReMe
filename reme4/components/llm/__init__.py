"""LLM model wrappers for AgentScope."""

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

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseLLM(BaseComponent):
    """Base wrapper for AgentScope chat models.

    Subclasses set ``credential_cls`` and inherit ``_start`` / ``_close``.
    """

    component_type = ComponentEnum.LLM
    credential_cls: type[CredentialBase]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: ChatModelBase | None = None

    async def _start(self) -> None:
        kwargs = dict(self.kwargs)
        credential = self.credential_cls(**kwargs.pop("credential", {}))
        model_cls = credential.get_chat_model_class()
        params_dict = kwargs.pop("parameters", None)
        parameters = model_cls.Parameters(**params_dict) if params_dict else None
        self.model = model_cls(credential=credential, parameters=parameters, **kwargs)

    async def _close(self) -> None:
        self.model = None


@R.register("openai")
class OpenAILLM(BaseLLM):
    """OpenAI chat model wrapper."""

    credential_cls = OpenAICredential


@R.register("anthropic")
class AnthropicLLM(BaseLLM):
    """Anthropic chat model wrapper."""

    credential_cls = AnthropicCredential


@R.register("dashscope")
class DashScopeLLM(BaseLLM):
    """DashScope chat model wrapper."""

    credential_cls = DashScopeCredential


@R.register("deepseek")
class DeepSeekLLM(BaseLLM):
    """DeepSeek chat model wrapper."""

    credential_cls = DeepSeekCredential


@R.register("gemini")
class GeminiLLM(BaseLLM):
    """Gemini chat model wrapper."""

    credential_cls = GeminiCredential


@R.register("moonshot")
class MoonshotLLM(BaseLLM):
    """Moonshot chat model wrapper."""

    credential_cls = MoonshotCredential


@R.register("ollama")
class OllamaLLM(BaseLLM):
    """Ollama chat model wrapper."""

    credential_cls = OllamaCredential


@R.register("xai")
class XAILLM(BaseLLM):
    """xAI chat model wrapper."""

    credential_cls = XAICredential


__all__ = [
    "BaseLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "DashScopeLLM",
    "DeepSeekLLM",
    "GeminiLLM",
    "MoonshotLLM",
    "OllamaLLM",
    "XAILLM",
]
