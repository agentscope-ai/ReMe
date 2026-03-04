"""Novita-compatible embedding model implementation."""

from typing import Literal
from .openai_embedding_model import OpenAIEmbeddingModel
from .openai_embedding_model_sync import OpenAIEmbeddingModelSync


class NovitaEmbeddingModel(OpenAIEmbeddingModel):
    """Asynchronous embedding model implementation compatible with Novita APIs."""

    def __init__(self, encoding_format: Literal["float", "base64"] = "float", **kwargs):
        """Initialize the Novita async embedding model, defaulting to Novita's API endpoint."""
        if "base_url" not in kwargs or kwargs["base_url"] is None:
            kwargs["base_url"] = "https://api.novita.ai/openai"
        super().__init__(encoding_format=encoding_format, **kwargs)


class NovitaEmbeddingModelSync(OpenAIEmbeddingModelSync):
    """Synchronous embedding model implementation compatible with Novita APIs."""

    def __init__(self, encoding_format: Literal["float", "base64"] = "float", **kwargs):
        """Initialize the Novita sync embedding model, defaulting to Novita's API endpoint."""
        if "base_url" not in kwargs or kwargs["base_url"] is None:
            kwargs["base_url"] = "https://api.novita.ai/openai"
        super().__init__(encoding_format=encoding_format, **kwargs)
