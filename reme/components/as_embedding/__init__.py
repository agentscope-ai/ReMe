"""AgentScope embedding model wrappers."""

import hashlib
import os
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
        """Return configured dimensions without forcing provider construction."""
        if self.model is not None:
            return self.model.dimensions
        dimensions = self.kwargs.get("dimensions")
        if dimensions is None:
            raise RuntimeError("Embedding dimensions are required before provider initialization.")
        return int(dimensions)

    @property
    def vector_space(self) -> tuple[str, ...]:
        """Return the fields that make vectors from two embedding setups incompatible.

        The wrapper identifies the provider consistently before and after lazy model
        construction.  Model details come from the live provider when one has been
        injected at runtime, otherwise they come from the configured kwargs.
        """
        if self.model is not None:
            return (
                self.backend or self.credential_cls.__name__,
                str(getattr(self.model, "model", self.kwargs.get("model") or "")),
                str(self.dimensions),
                self._model_endpoint(),
            )
        return (
            self.backend or self.credential_cls.__name__,
            str(self.kwargs.get("model") or ""),
            str(self.dimensions),
            self._configured_endpoint(),
        )

    @property
    def vector_space_id(self) -> str:
        """Return a short digest of :attr:`vector_space` for naming persisted vectors.

        Cache consumers use this digest to avoid reusing vectors produced by a
        different embedding setup.
        """
        return hashlib.sha256("\x1f".join(self.vector_space).encode()).hexdigest()[:12]

    @staticmethod
    def _endpoint(credential: Any) -> str:
        """Read the provider endpoint from a credential object or a raw kwargs dict."""
        for field in ("base_url", "host"):
            value = credential.get(field) if isinstance(credential, dict) else getattr(credential, field, None)
            if value:
                return str(value).rstrip("/")
        return ""

    def _configured_endpoint(self) -> str:
        """Resolve endpoint defaults without constructing the provider eagerly."""
        credential = self.kwargs.get("credential")
        if not isinstance(credential, dict):
            return self._endpoint(credential)
        fields = getattr(self.credential_cls, "model_fields", {})
        for name in ("base_url", "host"):
            value = credential.get(name)
            field = fields.get(name)
            if name not in credential and field is not None and not field.is_required():
                value = field.get_default(call_default_factory=True)
            if value:
                return str(value).rstrip("/")
        return ""

    def _model_endpoint(self) -> str:
        """Return the endpoint used by the constructed provider."""
        assert self.model is not None
        return self._endpoint(getattr(self.model, "credential", self.kwargs.get("credential")))

    async def __call__(self, inputs: list[Any], **kwargs) -> list[list[float]]:
        self._ensure_model()
        assert self.model is not None
        response = await self.model(inputs, **kwargs)  # pylint: disable=not-callable
        return response.embeddings

    async def _start(self) -> None:
        """Defer provider construction until the first remote embedding call."""
        return None

    def _ensure_model(self) -> None:
        """Construct the provider on demand while keeping dimensions locally available."""
        if self.model is not None:
            return

        kwargs = dict(self.kwargs)
        credential = self.credential_cls(**kwargs.pop("credential", {}))

        model_cls = self.credential_cls.get_embedding_model_class()
        if model_cls is None:
            raise ValueError(f"{self.credential_cls.__name__} does not support embeddings.")

        dimensions = self.dimensions
        kwargs.pop("dimensions", None)
        params_dict = kwargs.pop("parameters", None)
        parameters = model_cls.Parameters(**params_dict) if params_dict else None

        self.model = model_cls(
            credential=credential,
            dimensions=dimensions,
            parameters=parameters,
            **kwargs,
        )


@R.register("openai")
class OpenAIAsEmbedding(BaseAsEmbedding):
    """OpenAI embedding model wrapper."""

    credential_cls = OpenAICredential
    _default_base_url = "https://api.openai.com/v1"

    def _configured_endpoint(self) -> str:
        """Resolve the endpoint with the same precedence as the OpenAI SDK."""
        credential = self.kwargs.get("credential")
        base_url = credential.get("base_url") if isinstance(credential, dict) else None
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL") or self._default_base_url
        return str(base_url).rstrip("/")

    def _model_endpoint(self) -> str:
        """Read the endpoint the OpenAI client actually resolved."""
        assert self.model is not None
        client = getattr(self.model, "client", None)
        base_url = getattr(client, "base_url", None)
        return str(base_url).rstrip("/") if base_url is not None else super()._model_endpoint()


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
