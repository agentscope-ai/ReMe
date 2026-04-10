"""Module for registering AgentScope LLM models."""

import asyncio

from agentscope.model import OpenAIChatModel, ChatModelBase

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsLLM(BaseComponent):
    """Simple wrapper for AgentScope LLM models."""

    component_type = ComponentEnum.AS_LLM

    def __init__(self, **kwargs) -> None:
        """Initialize with model configuration."""
        super().__init__(**kwargs)
        self.model: ChatModelBase | None = None

    async def _start(self, app_context=None) -> None:
        """Initialize the AgentScope model instance."""

    async def _close(self) -> None:
        """Close the AgentScope model and release resources."""
        self.model = None


@R.register("openai")
class OpenAIAsLLM(BaseAsLLM):

    async def _start(self, app_context=None) -> None:
        """Initialize the AgentScope model instance."""
        self.model = OpenAIChatModel(**self.kwargs)

    async def _close(self) -> None:
        """Close the AgentScope model and release resources."""
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
