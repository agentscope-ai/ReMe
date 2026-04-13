"""Abstract base class for file storage backends."""

import re
from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ...enumeration import ComponentEnum
from ...schema import FileChunk, FileMetadata


class BaseFileStore(BaseComponent):
    """Abstract base class for file storage backends.

    Provides embedding resolution, validation, and safe embedding retrieval
    with automatic fallback on failure.
    """

    def __init__(
            self,
            store_name: str,
            db_path: str | Path,
            embedding_model: str = "default",
            fts_enabled: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._embedding_model_name: str = embedding_model
        self.embedding_model: BaseEmbeddingModel | None = None
        self.store_name: str = store_name
        self.db_path: Path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.vector_enabled: bool = bool(embedding_model)
        self.fts_enabled: bool = fts_enabled

        if not re.match(r"^[a-zA-Z0-9_]+$", store_name):
            raise ValueError(
                f"Invalid store name '{store_name}'. Only alphanumeric characters and underscores are allowed.")
        if not self.vector_enabled and not self.fts_enabled:
            raise ValueError("At least one of embedding_model or fts_enabled must be set.")

    async def _start(self, app_context=None):
        """Resolve embedding model from app_context."""
        if not self._embedding_model_name:
            return
        assert app_context is not None, "app_context must be provided"
        models = app_context.components.get(ComponentEnum.EMBEDDING_MODEL, {})
        if self._embedding_model_name not in models:
            raise ValueError(f"Embedding model '{self._embedding_model_name}' not found.")
        model = models[self._embedding_model_name]
        if not isinstance(model, BaseEmbeddingModel):
            raise TypeError(f"Expected BaseEmbeddingModel, got {type(model).__name__}")
        self.embedding_model = model

    async def _close(self):
        """Release embedding model reference."""
        self.embedding_model = None

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimensionality (default 1024)."""
        return self.embedding_model.dimensions if self.embedding_model else 1024

    def _disable_vector_search(self, reason: str = "embedding API error") -> None:
        """Disable vector search and log a warning."""
        if self.vector_enabled:
            self.logger.warning(f"[{self.store_name}] Disabling vector search: {reason}")
            self.vector_enabled = False

    async def _get_embeddings_safe(self, texts: list[str], **kwargs) -> list[list[float]] | None:
        """Get embeddings, returning None if vector search is disabled or an error occurs."""
        if not self.vector_enabled:
            return None
        try:
            assert self.embedding_model is not None, "Embedding model not initialized"
            return await self.embedding_model.get_embeddings(texts, **kwargs)
        except Exception as e:
            self._disable_vector_search(str(e))
            return None

    async def get_embedding(self, query: str, **kwargs) -> list[float] | None:
        """Get embedding for a single query string."""
        result = await self._get_embeddings_safe([query], **kwargs)
        return result[0] if result else None

    async def get_embeddings(self, queries: list[str], **kwargs) -> list[list[float]] | None:
        """Get embeddings for a batch of query strings."""
        return await self._get_embeddings_safe(queries, **kwargs)

    async def get_chunk_embedding(self, chunk: FileChunk, **kwargs) -> FileChunk:
        """Attach embedding to a single FileChunk."""
        chunk.embedding = await self.get_embedding(chunk.text, **kwargs)
        return chunk

    async def get_chunk_embeddings(self, chunks: list[FileChunk], **kwargs) -> list[FileChunk]:
        """Attach embeddings to a batch of FileChunk."""
        if not chunks:
            return chunks
        embeddings = await self.get_embeddings([c.text for c in chunks], **kwargs)
        if embeddings and len(embeddings) == len(chunks):
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
        else:
            for chunk in chunks:
                chunk.embedding = None
        return chunks

    @abstractmethod
    async def clear_all(self):
        """Clear all indexed data."""

    @abstractmethod
    async def upsert_file(self, file_meta: FileMetadata, chunks: list[FileChunk]):
        """Insert or update a file and its chunks."""

    @abstractmethod
    async def delete_file(self, path: str):
        """Delete a file and all its chunks."""

    @abstractmethod
    async def delete_file_chunks(self, path: str, chunk_ids: list[str]):
        """Delete specific chunks for a file."""

    @abstractmethod
    async def upsert_chunks(self, chunks: list[FileChunk]):
        """Insert or update specific chunks without affecting others."""

    @abstractmethod
    async def list_files(self) -> list[str]:
        """List all indexed file paths."""

    @abstractmethod
    async def get_file_metadata(self, path: str) -> FileMetadata | None:
        """Get file metadata."""

    @abstractmethod
    async def update_file_metadata(self, file_meta: FileMetadata) -> None:
        """Update file metadata without affecting chunks."""

    @abstractmethod
    async def get_file_chunks(self, path: str) -> list[FileChunk]:
        """Get all chunks for a file."""

    @abstractmethod
    async def vector_search(self, query: str, limit: int) -> list[FileChunk]:
        """Perform vector similarity search."""

    @abstractmethod
    async def keyword_search(self, query: str, limit: int) -> list[FileChunk]:
        """Perform full-text/keyword search."""

    @abstractmethod
    async def hybrid_search(
            self,
            query: str,
            limit: int,
            vector_weight: float = 0.7,
            candidate_multiplier: float = 3.0,
    ) -> list[FileChunk]:
        """Perform hybrid search combining vector and keyword results.

        Args:
            query: Search query text.
            limit: Maximum number of results.
            vector_weight: Weight for vector scores (0.0-1.0).
            candidate_multiplier: Multiplier for candidate pool size.

        Returns:
            FileChunk list with score populated, sorted by relevance.
        """
