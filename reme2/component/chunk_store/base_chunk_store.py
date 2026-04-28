"""Abstract base class for chunk storage backends."""

import re
from abc import abstractmethod
from pathlib import Path

from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ...enumeration import ComponentEnum
from ...schema import ChunkFilter, FileChunk


class BaseChunkStore(BaseComponent):
    """Abstract base class for chunk storage backends.

    Handles chunk persistence and retrieval (vector / keyword / hybrid search).
    File-level metadata and search filter resolution live in FileGraph; this
    layer only consumes a compiled ChunkFilter (path set) for restricting search.
    """

    component_type = ComponentEnum.CHUNK_STORE

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
                f"Invalid store name '{store_name}'. Only alphanumeric characters and underscores are allowed.",
            )
        if not self.vector_enabled and not self.fts_enabled:
            raise ValueError("At least one of embedding_model or fts_enabled must be set.")

    async def _start(self):
        if not self._embedding_model_name:
            return
        assert self.app_context is not None, "app_context must be provided"
        models = self.app_context.components.get(ComponentEnum.EMBEDDING_MODEL, {})
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

    # -- Keyword scoring utility --------------------------------------------

    @staticmethod
    def _score_keyword_match(query: str, text: str) -> float:
        """Score a keyword match using word-match ratio + phrase bonus."""
        words = query.split()
        if not words:
            return 0.0
        query_lower = query.lower()
        words_lower = [w.lower() for w in words]
        text_lower = text.lower()
        n_words = len(words)

        match_count = sum(1 for w in words_lower if w in text_lower)
        if match_count == 0:
            return 0.0

        base_score = match_count / n_words
        phrase_bonus = 0.2 if n_words > 1 and query_lower in text_lower else 0.0
        return min(1.0, base_score + phrase_bonus)

    # -- Hybrid search (concrete, delegates to abstract vector/keyword) -----

    async def hybrid_search(
            self,
            query: str,
            limit: int,
            vector_weight: float = 0.7,
            candidate_multiplier: float = 3.0,
            chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        """Perform hybrid search combining vector and keyword results."""
        assert 0.0 <= vector_weight <= 1.0

        candidates = min(200, max(1, int(limit * candidate_multiplier)))
        text_weight = 1.0 - vector_weight

        if self.vector_enabled and self.fts_enabled:
            keyword_results = await self.keyword_search(query, candidates, chunk_filter)
            vector_results = await self.vector_search(query, candidates, chunk_filter)

            if not keyword_results:
                return vector_results[:limit]
            if not vector_results:
                return keyword_results[:limit]

            merged = self._merge_hybrid_results(
                vector_results,
                keyword_results,
                vector_weight,
                text_weight,
            )
            return merged[:limit]
        elif self.vector_enabled:
            return await self.vector_search(query, limit, chunk_filter)
        elif self.fts_enabled:
            return await self.keyword_search(query, limit, chunk_filter)
        return []

    @staticmethod
    def _merge_hybrid_results(
            vector: list[FileChunk],
            keyword: list[FileChunk],
            vector_weight: float,
            text_weight: float,
    ) -> list[FileChunk]:
        """Merge vector and keyword results with weighted scoring."""
        merged: dict[str, FileChunk] = {}

        for result in vector:
            v_score = result.scores.get("vector", 0)
            result.scores["score"] = v_score * vector_weight
            merged[result.unique_key] = result

        for result in keyword:
            key = result.unique_key
            k_score = result.scores.get("keyword", 0)
            if key in merged:
                merged[key].scores["score"] += k_score * text_weight
            else:
                result.scores["score"] = k_score * text_weight
                merged[key] = result

        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # -- Filter utility -----------------------------------------------------

    @staticmethod
    def _apply_filter(chunks: list[FileChunk], chunk_filter: ChunkFilter | None) -> list[FileChunk]:
        if chunk_filter is None or chunk_filter.resolved_paths is None:
            return chunks
        return [c for c in chunks if chunk_filter.match_path(c.path)]

    # -- Abstract methods ---------------------------------------------------

    @abstractmethod
    async def clear_all(self):
        """Clear all indexed data."""

    @abstractmethod
    async def upsert_chunks(self, path: str, chunks: list[FileChunk]):
        """Insert or update all chunks for a file path."""

    @abstractmethod
    async def delete_chunks(self, path: str):
        """Delete all chunks for a file path."""

    @abstractmethod
    async def get_chunks(self, path: str) -> list[FileChunk]:
        """Get all chunks for a file path."""

    @abstractmethod
    async def vector_search(
            self,
            query: str,
            limit: int,
            chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        """Perform vector similarity search, optionally restricted by chunk_filter."""

    @abstractmethod
    async def keyword_search(
            self,
            query: str,
            limit: int,
            chunk_filter: ChunkFilter | None = None,
    ) -> list[FileChunk]:
        """Perform full-text/keyword search, optionally restricted by chunk_filter."""
