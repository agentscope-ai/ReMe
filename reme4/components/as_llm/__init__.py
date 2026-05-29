"""AgentScope LLM model wrappers."""

from agentscope.model import AnthropicChatModel, ChatModelBase, OpenAIChatModel

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsLLM(BaseComponent):
    """Base wrapper for AgentScope chat models. Builds ``self.model`` in ``_start``."""

    component_type = ComponentEnum.AS_LLM

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: ChatModelBase | None = None

    async def _close(self) -> None:
        self.model = None


@R.register("openai")
class OpenAIAsLLM(BaseAsLLM):
    """OpenAI chat model wrapper."""

    async def _start(self) -> None:
        kwargs = dict(self.kwargs)
        base_url = kwargs.pop("base_url", None)
        if base_url:
            client_kwargs = dict(kwargs.pop("client_kwargs", None) or {})
            client_kwargs.setdefault("base_url", base_url)
            kwargs["client_kwargs"] = client_kwargs
        self.model = OpenAIChatModel(**kwargs)

    async def _close(self) -> None:
        if self.model is not None:
            assert isinstance(self.model, OpenAIChatModel)
            await self.model.client.close()


@R.register("anthropic")
class AnthropicAsLLM(BaseAsLLM):
    """Anthropic chat model wrapper."""

    async def _start(self) -> None:
        self.model = AnthropicChatModel(**self.kwargs)

    async def _close(self) -> None:
        if self.model is not None:
            assert isinstance(self.model, AnthropicChatModel)
            await self.model.client.close()


__all__ = [
    "BaseAsLLM",
    "OpenAIAsLLM",
    "AnthropicAsLLM",
]
