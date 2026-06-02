"""AgentScope embedding model wrappers."""

from agentscope.embedding import (
    DashScopeMultiModalEmbedding as _AsDashScopeMultiModalEmbedding,
    DashScopeTextEmbedding,
    EmbeddingModelBase,
    GeminiTextEmbedding,
    OllamaTextEmbedding,
    OpenAITextEmbedding,
)

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseEmbedding(BaseComponent):
    """Base wrapper for AgentScope embedding models. Builds ``self.model`` in ``_start``."""

    component_type = ComponentEnum.EMBEDDING

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: EmbeddingModelBase | None = None

    @property
    def dimensions(self) -> int:
        assert self.model is not None
        return self.model.dimensions

    async def __call__(self, text: list[str], **kwargs) -> list[list[float]]:
        assert self.model is not None
        response = await self.model(text, **kwargs)
        return response.embeddings

    async def _close(self) -> None:
        self.model = None


@R.register("openai")
class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding model wrapper."""

    async def _start(self) -> None:
        self.model = OpenAITextEmbedding(**self.kwargs)

    async def _close(self) -> None:
        if self.model is not None:
            assert isinstance(self.model, OpenAITextEmbedding)
            await self.model.client.close()


@R.register("dashscope")
class DashScopeEmbedding(BaseEmbedding):
    """DashScope text embedding model wrapper."""

    async def _start(self) -> None:
        self.model = DashScopeTextEmbedding(**self.kwargs)


@R.register("dashscope_multimodal")
class DashScopeMultiModalEmbedding(BaseEmbedding):
    """DashScope multimodal embedding model wrapper."""

    async def _start(self) -> None:
        self.model = _AsDashScopeMultiModalEmbedding(**self.kwargs)


@R.register("gemini")
class GeminiEmbedding(BaseEmbedding):
    """Gemini embedding model wrapper."""

    async def _start(self) -> None:
        self.model = GeminiTextEmbedding(**self.kwargs)


@R.register("ollama")
class OllamaEmbedding(BaseEmbedding):
    """Ollama embedding model wrapper."""

    async def _start(self) -> None:
        self.model = OllamaTextEmbedding(**self.kwargs)


__all__ = [
    "BaseEmbedding",
    "OpenAIEmbedding",
    "DashScopeEmbedding",
    "DashScopeMultiModalEmbedding",
    "GeminiEmbedding",
    "OllamaEmbedding",
]
