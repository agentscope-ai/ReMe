"""AgentScope embedding model wrappers."""

from typing import Any

from agentscope.credential import (
    CredentialBase,
    DashScopeCredential,
    GeminiCredential,
    OllamaCredential,
    OpenAICredential,
)
from agentscope.embedding import (
    EmbeddingModelBase,
)

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsEmbedding(BaseComponent):
    """Base wrapper for AgentScope embedding models."""

    component_type = ComponentEnum.AS_EMBEDDING
    credential_cls: type[CredentialBase]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model: EmbeddingModelBase[Any] | None = None

    @property
    def dimensions(self) -> int:
        """Return the embedding dimension size."""
        assert self.model is not None
        return self.model.dimensions

    async def __call__(self, inputs: list[Any], **kwargs) -> list[list[float]]:
        assert self.model is not None
        response = await self.model(inputs, **kwargs)  # pylint: disable=not-callable
        return response.embeddings

    async def _start(self) -> None:
        if self.model is not None:
            return

        kwargs = dict(self.kwargs)
        credential = self.credential_cls(**kwargs.pop("credential", {}))

        model_cls = self.credential_cls.get_embedding_model_class()
        if model_cls is None:
            raise ValueError(f"{self.credential_cls.__name__} does not support embeddings.")

        params_dict = kwargs.pop("parameters", None)
        parameters = model_cls.Parameters(**params_dict) if params_dict else None
        self.model = model_cls(credential=credential, parameters=parameters, **kwargs)


@R.register("openai")
class OpenAIAsEmbedding(BaseAsEmbedding):
    """OpenAI embedding model wrapper."""

    credential_cls = OpenAICredential


@R.register("dashscope")
class DashScopeAsEmbedding(BaseAsEmbedding):
    """DashScope embedding model wrapper."""

    credential_cls = DashScopeCredential


@R.register("dashscope_multimodal")
class DashScopeMultiModalAsEmbedding(BaseAsEmbedding):
    """DashScope multimodal embedding model wrapper."""

    credential_cls = DashScopeCredential


@R.register("gemini")
class GeminiAsEmbedding(BaseAsEmbedding):
    """Gemini embedding model wrapper."""

    credential_cls = GeminiCredential


@R.register("ollama")
class OllamaAsEmbedding(BaseAsEmbedding):
    """Ollama embedding model wrapper."""

    credential_cls = OllamaCredential


__all__ = [
    "BaseAsEmbedding",
    "OpenAIAsEmbedding",
    "DashScopeAsEmbedding",
    "DashScopeMultiModalAsEmbedding",
    "GeminiAsEmbedding",
    "OllamaAsEmbedding",
]
