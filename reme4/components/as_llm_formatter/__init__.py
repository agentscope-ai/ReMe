"""AgentScope LLM formatter wrappers."""

from agentscope.formatter import AnthropicChatFormatter, FormatterBase

from .reme_openai_chat_formatter import ReMeOpenAIChatFormatter
from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsLLMFormatter(BaseComponent):
    """Base wrapper for AgentScope formatters. Builds ``self.formatter`` in ``_start``."""

    component_type = ComponentEnum.AS_LLM_FORMATTER

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.formatter: FormatterBase | None = None

    async def _close(self) -> None:
        self.formatter = None


@R.register("openai")
class AsOpenAIChatFormatter(BaseAsLLMFormatter):
    """OpenAI chat formatter wrapper (uses ReMe extensions)."""

    async def _start(self) -> None:
        self.formatter = ReMeOpenAIChatFormatter(**self.kwargs)


@R.register("anthropic")
class AsAnthropicChatFormatter(BaseAsLLMFormatter):
    """Anthropic chat formatter wrapper."""

    async def _start(self) -> None:
        self.formatter = AnthropicChatFormatter(**self.kwargs)


__all__ = [
    "BaseAsLLMFormatter",
    "AsOpenAIChatFormatter",
    "AsAnthropicChatFormatter",
]
