"""Asynchronous OpenAI-compatible embedding model implementation for ReMe."""

from typing import Literal

from openai import AsyncOpenAI

from .base_embedding_model import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Asynchronous embedding model implementation compatible with OpenAI-style APIs."""

    def __init__(self, encoding_format: Literal["float", "base64"] = "float", **kwargs):
        """Initialize the OpenAI async embedding model with API credentials and configuration."""
        super().__init__(**kwargs)
        self.encoding_format: Literal["float", "base64"] = encoding_format

        # Lazy client initialization
        self._client = None

    def _create_client(self):
        """Create and return an internal AsyncOpenAI client instance."""
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def client(self):
        """Lazily create and return the AsyncOpenAI client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    async def _get_embeddings(self, input_text: list[str], **kwargs) -> list[list[float]]:
        """Fetch embeddings from the API for a batch of strings."""
        completion = await self.client.embeddings.create(
            model=self.model_name,
            input=input_text,
            dimensions=self.dimensions,
            encoding_format=self.encoding_format,
            **self.kwargs,
            **kwargs,
        )

        result_emb = [[] for _ in range(len(input_text))]
        for emb in completion.data:
            result_emb[emb.index] = emb.embedding
        return result_emb

    async def close(self):
        """Close the asynchronous OpenAI client and release network resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        await super().close()
