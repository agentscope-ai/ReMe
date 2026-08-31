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
    "AnthropicAsLLM",
    "DashScopeAsLLM",
    "DeepSeekAsLLM",
    "GeminiAsLLM",
    "MoonshotAsLLM",
    "OllamaAsLLM",
    "XAIAsLLM",
]
