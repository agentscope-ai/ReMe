"""Base storage interface for memory manager."""

import asyncio
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable

from ..embedding import BaseEmbeddingModel
from ..enumeration import MemorySource
from ..schema import FileMetadata, MemoryChunk, MemorySearchResult


class BaseMemoryStore(ABC):
    """Abstract base class for memory storage backends."""

    def __init__(
        self,
        store_name: str,
        db_path: str | Path,
        thread_pool: ThreadPoolExecutor,
        embedding_model: BaseEmbeddingModel,
        vector_enabled: bool = False,
        fts_enabled: bool = True,
        **kwargs,
    ):
        """Initialize"""
        # Validate store_name to prevent SQL injection
        # Only allow alphanumeric characters and underscores
        if not re.match(r"^[a-zA-Z0-9_]+$", store_name):
            raise ValueError(f"Invalid '{store_name}'. Only alphanumeric characters and underscores are allowed.")

        # Ensure at least one search method is enabled
        if not vector_enabled and not fts_enabled:
            raise ValueError("At least one of vector_enabled or fts_enabled must be True.")

        self.store_name: str = store_name
        self.db_path: Path = Path(db_path)
        self.thread_pool: ThreadPoolExecutor = thread_pool
        self.embedding_model: BaseEmbeddingModel = embedding_model
        self.vector_enabled: bool = vector_enabled
        self.fts_enabled: bool = fts_enabled
        self.kwargs: dict = kwargs

    @property
    def embedding_dim(self) -> int:
        """Get the embedding model's dimensionality."""
        return self.embedding_model.dimensions

    def _get_mock_embedding(self) -> list[float]:
        """Generate a zero vector based on embedding model dimensions."""
        return [0.0] * self.embedding_dim

    async def get_embedding(self, query: str, **kwargs) -> list[float]:
        """Get embedding for a single query string."""
        if not self.vector_enabled:
            return self._get_mock_embedding()
        return await self.embedding_model.get_embedding(query, **kwargs)

    async def get_embeddings(self, queries: list[str], **kwargs) -> list[list[float]]:
        """Get embeddings for a batch of query strings."""
        if not self.vector_enabled:
            return [self._get_mock_embedding() for _ in queries]
        return await self.embedding_model.get_embeddings(queries, **kwargs)

    async def get_chunk_embedding(self, chunk: MemoryChunk, **kwargs) -> MemoryChunk:
        """Generate and populate embedding field for a single MemoryChunk object."""
        if not self.vector_enabled:
            chunk.embedding = self._get_mock_embedding()
            return chunk
        return await self.embedding_model.get_chunk_embedding(chunk, **kwargs)

    async def get_chunk_embeddings(self, chunks: list[MemoryChunk], **kwargs) -> list[MemoryChunk]:
        """Generate and populate embedding fields for a batch of MemoryChunk objects."""
        if not self.vector_enabled:
            mock_embedding = self._get_mock_embedding()
            for chunk in chunks:
                chunk.embedding = mock_embedding.copy()
            return chunks
        return await self.embedding_model.get_chunk_embeddings(chunks, **kwargs)

    async def _run_sync_in_executor(self, sync_func: Callable, *args, **kwargs):
        """Run a synchronous function in the context-defined thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.thread_pool, partial(sync_func, *args, **kwargs))  # noqa

    @abstractmethod
    async def start(self):
        """Initialize the storage backend."""

    @abstractmethod
    async def upsert_file(self, file_meta: FileMetadata, source: MemorySource, chunks: list[MemoryChunk]):
        """Insert or update a file and its chunks."""

    @abstractmethod
    async def delete_file(self, path: str, source: MemorySource):
        """Delete a file and all its chunks."""

    @abstractmethod
    async def delete_file_chunks(self, path: str, chunk_ids: list[str]):
        """Delete chunks for a file."""

    @abstractmethod
    async def upsert_chunks(self, chunks: list[MemoryChunk], source: MemorySource):
        """Insert or update specific chunks without affecting other chunks."""

    @abstractmethod
    async def list_files(self, source: MemorySource) -> list[str]:
        """List all indexed file paths for a source."""

    @abstractmethod
    async def get_file_metadata(self, path: str, source: MemorySource) -> FileMetadata | None:
        """Get full file metadata with statistics."""

    @abstractmethod
    async def get_file_chunks(self, path: str, source: MemorySource) -> list[MemoryChunk]:
        """Get all chunks for a file."""

    @abstractmethod
    async def vector_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform vector similarity search.

        Args:
            query: Query embedding vector
            limit: Maximum number of results
            sources: Optional list of sources to filter

        Returns:
            List of search results sorted by similarity
        """

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform keyword/full-text search.

        Args:
            query: Search query text
            limit: Maximum number of results
            sources: Optional list of sources to filter

        Returns:
            List of search results sorted by relevance
        """

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
    ) -> list[MemorySearchResult]:
        """Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query text
            limit: Maximum number of results
            sources: Optional list of sources to filter
            vector_weight: Weight for vector search results (0.0-1.0).
                          Keyword weight = 1.0 - vector_weight.
            candidate_multiplier: Multiplier for candidate pool size.
                          candidates = limit * candidate_multiplier

        Returns:
            List of search results sorted by combined relevance score
        """

    @abstractmethod
    async def clear_all(self):
        """Clear all indexed data."""

    @abstractmethod
    async def close(self):
        """Close storage and release resources."""
