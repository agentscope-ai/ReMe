"""Module for registering AgentScope LLM formatters."""

from agentscope.formatter import OpenAIChatFormatter

from .reme_openai_chat_formatter import ReMeOpenAIChatFormatter
from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsLLMFormatter(BaseComponent):
    """Base wrapper for AgentScope LLM formatters."""

    component_type = ComponentEnum.AS_LLM_FORMATTER

    def __init__(self, **kwargs) -> None:
        """Initialize with formatter configuration."""
        super().__init__(**kwargs)
        self.formatter: OpenAIChatFormatter | None = None

    async def _start(self, app_context=None) -> None:
        """Initialize the formatter instance."""

    async def _close(self) -> None:
        """Close the formatter (no-op for formatter)."""
        self.formatter = None


@R.register("openai")
class AsOpenAIChatFormatter(BaseAsLLMFormatter):
    """Wrapper for ReMeOpenAIChatFormatter."""

    async def _start(self, app_context=None) -> None:
        """Initialize the formatter instance."""
        self.formatter = ReMeOpenAIChatFormatter(**self.kwargs)


__all__ = [
    "BaseAsLLMFormatter",
    "AsOpenAIChatFormatter",
]
