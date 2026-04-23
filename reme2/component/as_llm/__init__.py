"""AgentScope LLM model wrappers."""

from agentscope.model import OpenAIChatModel, ChatModelBase, AnthropicChatModel

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

    async def _start(self) -> None:
        """Initialize the model."""

    async def _close(self) -> None:
        """Release model resources."""
        self.model = None


@R.register("openai")
class OpenAIAsLLM(BaseAsLLM):
    """OpenAI chat model wrapper."""

    async def _start(self) -> None:
        """Initialize the OpenAI chat model."""
        self.model = OpenAIChatModel(**self.kwargs)

    async def _close(self) -> None:
        """Close the HTTP client and release resources."""
        if self.model is not None:
            assert isinstance(self.model, OpenAIChatModel)
            await self.model.client.close()


@R.register("anthropic")
class AnthropicAsLLM(BaseAsLLM):
    """Anthropic chat model wrapper."""

    async def _start(self) -> None:
        """Initialize the Anthropic chat model."""
        self.model = AnthropicChatModel(**self.kwargs)

    async def _close(self) -> None:
        """Close the HTTP client and release resources."""
        if self.model is not None:
            assert isinstance(self.model, AnthropicChatModel)
            await self.model.client.close()


__all__ = [
    "BaseAsLLM",
    "OpenAIAsLLM",
    "AnthropicAsLLM",
]
