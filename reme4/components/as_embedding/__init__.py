"""AgentScope embedding model wrappers."""

from agentscope.embedding import (
    DashScopeMultiModalEmbedding,
    DashScopeTextEmbedding,
    EmbeddingModelBase,
    GeminiTextEmbedding,
    OllamaTextEmbedding,
    OpenAITextEmbedding,
)

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsEmbedding(BaseComponent):
    """Base wrapper for AgentScope embedding models. Builds ``self.model`` in ``_start``."""

    component_type = ComponentEnum.AS_EMBEDDING

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: EmbeddingModelBase | None = None

    @property
    def dimensions(self) -> int:
        """Return the embedding dimension size."""
        assert self.model is not None
        return self.model.dimensions

    async def __call__(self, text: list[str], **kwargs) -> list[list[float]]:
        assert self.model is not None
        response = await self.model(text, **kwargs)  # pylint: disable=not-callable
        return response.embeddings

    async def _close(self) -> None:
        self.model = None


@R.register("openai")
class OpenAIAsEmbedding(BaseAsEmbedding):
    """OpenAI embedding model wrapper."""

    async def _start(self) -> None:
        self.model = OpenAITextEmbedding(**self.kwargs)

    async def _close(self) -> None:
        if self.model is not None:
            assert isinstance(self.model, OpenAITextEmbedding)
            await self.model.client.close()


@R.register("dashscope")
class DashScopeAsEmbedding(BaseAsEmbedding):
    """DashScope text embedding model wrapper."""

    async def _start(self) -> None:
        self.model = DashScopeTextEmbedding(**self.kwargs)


@R.register("dashscope_multimodal")
class DashScopeMultiModalAsEmbedding(BaseAsEmbedding):
    """DashScope multimodal embedding model wrapper."""

    async def _start(self) -> None:
        self.model = DashScopeMultiModalEmbedding(**self.kwargs)


@R.register("gemini")
class GeminiAsEmbedding(BaseAsEmbedding):
    """Gemini embedding model wrapper."""

    async def _start(self) -> None:
        self.model = GeminiTextEmbedding(**self.kwargs)


@R.register("ollama")
class OllamaAsEmbedding(BaseAsEmbedding):
    """Ollama embedding model wrapper."""

    async def _start(self) -> None:
        self.model = OllamaTextEmbedding(**self.kwargs)


__all__ = [
    "BaseAsEmbedding",
    "OpenAIAsEmbedding",
    "DashScopeAsEmbedding",
    "DashScopeMultiModalAsEmbedding",
    "GeminiAsEmbedding",
    "OllamaAsEmbedding",
]
