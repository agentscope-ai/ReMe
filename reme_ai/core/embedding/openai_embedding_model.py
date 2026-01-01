"""Asynchronous OpenAI-compatible embedding model implementation for ReMe."""

import os
from typing import Literal

from openai import AsyncOpenAI

from .base_embedding_model import BaseEmbeddingModel
from ..context import C


@C.register_embedding_model("openai")
class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Asynchronous embedding model implementation compatible with OpenAI-style APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        encoding_format: Literal["float", "base64"] = "float",
        **kwargs,
    ):
        """Initialize the OpenAI async embedding model with API credentials and configuration."""
        super().__init__(**kwargs)
        self.api_key: str = api_key or os.getenv("REME_EMBEDDING_API_KEY", "")
        self.base_url: str = base_url or os.getenv("REME_EMBEDDING_BASE_URL", "")
        self.encoding_format: Literal["float", "base64"] = encoding_format

        # Create client using factory method
        self._client = self._create_client()

    def _create_client(self):
        """Create and return an internal AsyncOpenAI client instance."""
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def _get_embeddings(self, input_text: list[str]) -> list[list[float]]:
        """Fetch embeddings from the API for a batch of strings."""
        completion = await self._client.embeddings.create(
            model=self.model_name,
            input=input_text,
            dimensions=self.dimensions,
            encoding_format=self.encoding_format,
        )

        result_emb = [[] for _ in range(len(input_text))]
        for emb in completion.data:
            result_emb[emb.index] = emb.embedding
        return result_emb

    async def close(self):
        """Close the asynchronous OpenAI client and release network resources."""
        await self._client.close()
