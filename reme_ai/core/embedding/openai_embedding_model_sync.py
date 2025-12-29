"""OpenAI sync embedding model implementation for ReMe.

This module provides a synchronous OpenAI-compatible embedding model implementation that works with
OpenAI-compatible embedding APIs, including OpenAI's official API and
other services that follow the same interface.

For asynchronous operations, use OpenAIEmbeddingModel from openai_embedding_model module.
"""

from typing import List

from openai import OpenAI

from .openai_embedding_model import OpenAIEmbeddingModel
from ..context import C


@C.register_embedding_model("openai_sync")
class OpenAIEmbeddingModelSync(OpenAIEmbeddingModel):
    """
    OpenAI-compatible sync embedding model implementation.

    This class inherits from OpenAIEmbeddingModel and provides synchronous versions of
    the embedding methods. It overrides _create_client() to create a synchronous OpenAI
    client instead of async.

    For asynchronous operations, use OpenAIEmbeddingModel from openai_embedding_model module.

    Attributes:
        _client: Synchronous OpenAI client instance

    Example:
        >>> embedding_model = OpenAIEmbeddingModelSync(
        ...     model_name="text-embedding-v4",
        ...     api_key="sk-...",
        ...     dimensions=1024
        ... )
        >>> embeddings = embedding_model.get_embeddings_sync(["Hello world"])
    """

    def _create_client(self):
        """
        Create and return the synchronous OpenAI client instance.

        This method overrides the parent class method to provide a synchronous
        client instead of an asynchronous one.

        Returns:
            OpenAI client instance for sync operations
        """
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _get_embeddings_sync(self, input_text: str | List[str]) -> List[List[float]] | List[float]:
        """
        Get embeddings synchronously from the OpenAI-compatible API.

        This method implements the abstract _get_embeddings_sync method from BaseEmbeddingModel
        by calling the OpenAI-compatible embeddings API synchronously.

        Args:
            input_text: Single text string or list of text strings to embed

        Returns:
            Single embedding vector (List[float]) if input is str,
            or list of embedding vectors (List[List[float]]) if input is List[str]

        Raises:
            RuntimeError: If unsupported input type is provided
        """
        completion = self._client.embeddings.create(
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

    def close_sync(self):
        """
        Close the synchronous OpenAI client and release resources.

        This method properly closes the HTTP connection pool used by the
        synchronous OpenAI client. It should be called when the embedding
        model instance is no longer needed to avoid resource leaks.

        Example:
            >>> embedding_model = OpenAIEmbeddingModelSync(model_name="text-embedding-v4")
            >>> # ... use the embedding model ...
            >>> embedding_model.close_sync()
        """
        self._client.close()
