"""Module for registering AgentScope LLM formatters."""

from agentscope.formatter import OpenAIChatFormatter

from .reme_openai_chat_formatter import ReMeOpenAIChatFormatter
from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


@R.register("openai")
class AsOpenAIChatFormatter(BaseComponent):
    """Wrapper for ReMeOpenAIChatFormatter."""

    component_type = ComponentEnum.AS_LLM_FORMATTER

    def __init__(self, **kwargs) -> None:
        """Initialize with formatter configuration."""
        super().__init__(**kwargs)
        self.formatter: OpenAIChatFormatter | None = None

    async def _start(self, app_context=None) -> None:
        """Initialize the formatter instance."""
        self.formatter = ReMeOpenAIChatFormatter(**self.kwargs)

    async def _close(self) -> None:
        """Close the formatter (no-op for formatter)."""
        self.formatter = None


__all__ = [
    "AsOpenAIChatFormatter",
]
