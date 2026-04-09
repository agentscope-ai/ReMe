"""Module for registering AgentScope LLM models."""

from agentscope.model import OpenAIChatModel

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


@R.register("openai")
class AsOpenAIChatModel(BaseComponent):
    """Simple wrapper for AgentScope LLM models."""

    component_type = ComponentEnum.AS_LLM

    def __init__(self, **kwargs) -> None:
        """Initialize with model configuration."""
        super().__init__(**kwargs)
        self.model: OpenAIChatModel | None = None

    async def _start(self, app_context=None) -> None:
        """Initialize the AgentScope model instance."""
        self.model = OpenAIChatModel(**self.kwargs)

    async def _close(self) -> None:
        """Close the AgentScope model and release resources."""
        if self.model is not None:
            await self.model.client.close()
            self.model = None


__all__ = [
    "AsOpenAIChatModel",
]
