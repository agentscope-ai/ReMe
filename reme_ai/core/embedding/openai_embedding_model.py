"""OpenAI async embedding model implementation for ReMe.

This module provides an async OpenAI-compatible embedding model implementation that works with
OpenAI-compatible embedding APIs, including OpenAI's official API and
other services that follow the same interface.

For synchronous operations, use OpenAIEmbeddingModelSync from openai_embedding_model_sync module.
"""

import os
from typing import Literal, List, Optional

from openai import AsyncOpenAI

from .base_embedding_model import BaseEmbeddingModel
from ..context import C


@C.register_embedding_model("openai")
class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    OpenAI-compatible async embedding model implementation.

    This class provides async integration with OpenAI's Embeddings API or
    any compatible API endpoint (e.g., Azure OpenAI, local models with
    OpenAI-compatible servers).

    For synchronous operations, use OpenAIEmbeddingModelSync from openai_embedding_model_sync module.

    The API key and base URL can be provided either as constructor arguments
    or through environment variables (FLOW_EMBEDDING_API_KEY and FLOW_EMBEDDING_BASE_URL).

    Attributes:
        api_key: OpenAI API key or compatible service key
        base_url: Base URL for the API endpoint
        encoding_format: Encoding format for embeddings ("float" or "base64")
        _client: Asynchronous OpenAI client instance

    Example:
        >>> embedding_model = OpenAIEmbeddingModel(
        ...     model_name="text-embedding-v4",
        ...     api_key="sk-...",
        ...     dimensions=1024
        ... )
        >>> embeddings = await embedding_model.get_embeddings(["Hello world"])
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            encoding_format: Literal["float", "base64"] = "float",
            **kwargs,
    ):
        """
        Initialize the OpenAI async embedding model.

        Args:
            api_key: API key for authentication (defaults to FLOW_EMBEDDING_API_KEY env var)
            base_url: Base URL for the API endpoint (defaults to FLOW_EMBEDDING_BASE_URL env var)
            encoding_format: Encoding format for embeddings ("float" or "base64")
            **kwargs: Additional arguments passed to BaseEmbeddingModel, including:
                - model_name: Name of the model to use (required)
                - dimensions: Dimensionality of the embedding vectors (default: 1024)
                - max_batch_size: Maximum batch size for processing (default: 10)
                - max_retries: Maximum retry attempts on failure (default: 3)
                - raise_exception: Whether to raise exception on failure (default: True)
        """
        super().__init__(**kwargs)
        self.api_key: str = api_key or os.getenv("REME_EMBEDDING_API_KEY", "")
        self.base_url: str = base_url or os.getenv("REME_EMBEDDING_BASE_URL", "")
        self.encoding_format: Literal["float", "base64"] = encoding_format

        # Create client using factory method
        self._client = self._create_client()

    def _create_client(self):
        """
        Create and return the OpenAI client instance.

        This method can be overridden by subclasses to provide different client implementations
        (e.g., synchronous vs asynchronous clients).

        Returns:
            AsyncOpenAI client instance for async operations
        """
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def _get_embeddings(self, input_text: str | List[str]) -> List[List[float]] | List[float]:
        """
        Get embeddings asynchronously from the OpenAI-compatible API.

        This method implements the abstract _get_embeddings method from BaseEmbeddingModel
        by calling the OpenAI-compatible embeddings API asynchronously.

        Args:
            input_text: Single text string or list of text strings to embed

        Returns:
            Single embedding vector (List[float]) if input is str,
            or list of embedding vectors (List[List[float]]) if input is List[str]

        Raises:
            RuntimeError: If unsupported input type is provided
        """
        completion = await self._client.embeddings.create(
            model=self.model_name,
            input=input_text,
            dimensions=self.dimensions,
            encoding_format=self.encoding_format,
        )

        if isinstance(input_text, str):
            return completion.data[0].embedding

        elif isinstance(input_text, list):
            result_emb = [[] for _ in range(len(input_text))]
            for emb in completion.data:
                result_emb[emb.index] = emb.embedding
            return result_emb

        else:
            raise RuntimeError(f"unsupported type={type(input_text)}")

    async def close(self):
        """
        Asynchronously close the async OpenAI client and release resources.

        This method properly closes the HTTP connection pool used by the
        asynchronous OpenAI client. It should be called when the embedding
        model instance is no longer needed to avoid resource leaks.

        Example:
            >>> embedding_model = OpenAIEmbeddingModel(model_name="text-embedding-v4")
            >>> # ... use the embedding model asynchronously ...
            >>> await embedding_model.close()
        """
        await self._client.close()
