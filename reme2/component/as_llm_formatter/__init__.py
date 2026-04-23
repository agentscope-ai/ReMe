"""Module for AgentScope LLM formatter components."""

from agentscope.formatter import FormatterBase, AnthropicChatFormatter

from .reme_openai_chat_formatter import ReMeOpenAIChatFormatter
from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsLLMFormatter(BaseComponent):
    """Base wrapper for AgentScope LLM formatters."""

    component_type = ComponentEnum.AS_LLM_FORMATTER

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.formatter: FormatterBase | None = None

    async def _start(self) -> None:
        """Initialize the formatter."""

    async def _close(self) -> None:
        self.formatter = None


@R.register("openai")
class AsOpenAIChatFormatter(BaseAsLLMFormatter):
    """Wrapper for OpenAI chat completion formatter."""

    async def _start(self) -> None:
        """Initialize the OpenAI chat formatter."""
        self.formatter = ReMeOpenAIChatFormatter(**self.kwargs)


@R.register("anthropic")
class AsAnthropicChatFormatter(BaseAsLLMFormatter):
    """Wrapper for Anthropic chat completion formatter."""

    async def _start(self) -> None:
        self.formatter = AnthropicChatFormatter(**self.kwargs)


__all__ = [
    "BaseAsLLMFormatter",
    "AsOpenAIChatFormatter",
    "AsAnthropicChatFormatter",
]
