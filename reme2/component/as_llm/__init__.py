"""AgentScope LLM model wrappers."""

import asyncio

from agentscope.model import OpenAIChatModel, ChatModelBase

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsLLM(BaseComponent):
    """Base wrapper for AgentScope LLM models.

    Subclasses should implement _start() to initialize self.model.
    """

    component_type = ComponentEnum.AS_LLM

    def __init__(self, **kwargs) -> None:
        """Initialize with model configuration kwargs."""
        super().__init__(**kwargs)
        self.model: ChatModelBase | None = None

    async def _start(self, app_context=None) -> None:
        """Initialize the AgentScope model. Override in subclasses."""

    async def _close(self) -> None:
        """Release model resources."""
        self.model = None


@R.register("openai")
class OpenAIAsLLM(BaseAsLLM):
    """OpenAI chat model wrapper."""

    async def _start(self, app_context=None) -> None:
        """Initialize the OpenAI chat model."""
        self.model = OpenAIChatModel(**self.kwargs)

    async def _close(self) -> None:
        """Close the HTTP client and release resources."""
        if self.model is not None:
            client = getattr(self.model, "client", None)
            if client is not None and hasattr(client, "close"):
                close_method = client.close
                if asyncio.iscoroutinefunction(close_method):
                    await close_method()
                else:
                    close_method()
            self.model = None


__all__ = [
    "BaseAsLLM",
    "OpenAIAsLLM",
]
