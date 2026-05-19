"""Zvec storage backend for file store."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .base_file_store import BaseFileStore
from ..enumeration import MemorySource
from ..schema import FileMetadata, MemoryChunk, MemorySearchResult
from ..utils import get_logger

logger = get_logger()

_ZVEC_IMPORT_ERROR: Exception | None = None

try:
    import zvec  # type: ignore[import-untyped]
    from zvec import (
        CollectionOption,
        CollectionSchema,
        DataType,
        Doc,
        FieldSchema,
        HnswIndexParam,
        InvertIndexParam,
        VectorQuery,
        VectorSchema,
    )
    from zvec.typing import MetricType
except Exception as e:
    _ZVEC_IMPORT_ERROR = e
    zvec = None  # type: ignore[assignment]


# zvec max topk (will be lifted to 100,000 in zvec v0.3.2+)
_ZVEC_MAX_TOPK = 1024

# Default vector field name
_DEFAULT_VECTOR_FIELD = "embedding"


def _escape(value: str) -> str:
    """Escape a string value for zvec filter expressions."""
    return value.replace("'", "\\'")


def _build_file_store_schema(name: str, dimension: int) -> CollectionSchema:
    """Build a zvec CollectionSchema for file store chunks."""
    return CollectionSchema(
        name=name,
        fields=[
            FieldSchema("content", DataType.STRING, nullable=True, index_param=InvertIndexParam()),
            FieldSchema("path", DataType.STRING, nullable=True, index_param=InvertIndexParam()),
            FieldSchema("source", DataType.STRING, nullable=True, index_param=InvertIndexParam()),
            FieldSchema("start_line", DataType.INT64, nullable=True),
            FieldSchema("end_line", DataType.INT64, nullable=True),
            FieldSchema("hash", DataType.STRING, nullable=True),
            FieldSchema("updated_at", DataType.INT64, nullable=True),
            FieldSchema("file_metadata", DataType.STRING, nullable=True),
        ],
        vectors=[
            VectorSchema(
                name=_DEFAULT_VECTOR_FIELD,
                data_type=DataType.VECTOR_FP32,
                dimension=dimension,
                index_param=HnswIndexParam(metric_type=MetricType.COSINE),
            ),
        ],
    )


def _chunk_to_doc(chunk: MemoryChunk, file_meta_json: str = "{}") -> Doc:
    """Convert a MemoryChunk to a zvec Doc."""
    fields: dict[str, Any] = {
        "content": chunk.text,
        "path": chunk.path,
        "source": chunk.source.value if chunk.source else "",
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "hash": chunk.hash,
        "updated_at": int(time.time() * 1000),
        "file_metadata": file_meta_json,
    }
    vectors: dict[str, Any] = {}
    if chunk.embedding is not None:
        vectors[_DEFAULT_VECTOR_FIELD] = chunk.embedding
    return Doc(id=chunk.id, fields=fields, vectors=vectors)


def _doc_to_chunk(doc: Doc) -> MemoryChunk:
    """Convert a zvec Doc to a MemoryChunk."""
    raw_vector = doc.vector(_DEFAULT_VECTOR_FIELD)
    vector = raw_vector if isinstance(raw_vector, list) and len(raw_vector) > 0 else None
    return MemoryChunk(
        id=str(doc.id),
        path=str(doc.field("path") or ""),
        source=MemorySource(str(doc.field("source") or "")),
        start_line=int(doc.field("start_line") or 0),
        end_line=int(doc.field("end_line") or 0),
        text=str(doc.field("content") or ""),
        hash=str(doc.field("hash") or ""),
        embedding=vector,
    )


def _build_source_filter(sources: list[MemorySource] | None) -> str | None:
    """Build a zvec filter expression for source filtering."""
    if not sources:
        return None
    if len(sources) == 1:
        return f"source='{_escape(sources[0].value)}'"
    vals = ", ".join(f"'{_escape(s.value)}'" for s in sources)
    return f"source IN ({vals})"


class ZvecFileStore(BaseFileStore):
    """Zvec file storage with vector and keyword search.

    Provides zvec-backed persistent storage with:
    - Vector similarity search (native zvec HNSW)
    - Keyword search (Python substring matching on fetched results)
    - Hybrid search (weighted fusion of vector and keyword results)

    Note:
        Keyword search operates on chunks fetched from zvec, which is subject
        to the topk limit (1024 in zvec < v0.3.2, 100,000 in v0.3.2+).
        For collections with more chunks than the topk limit, keyword search
        may not scan all documents.
    """

    def __init__(
        self,
        store_name: str,
        db_path: str | Path,
        embedding_model: Any | None = None,
        vector_enabled: bool = False,
        fts_enabled: bool = True,
        dimension: int = 1024,
        **kwargs: Any,
    ):
        if _ZVEC_IMPORT_ERROR is not None:
            raise ImportError(
                "Zvec requires extra dependencies. Install with `pip install zvec`",
            ) from _ZVEC_IMPORT_ERROR

        super().__init__(
            store_name=store_name,
            db_path=db_path,
            embedding_model=embedding_model,
            vector_enabled=vector_enabled,
            fts_enabled=fts_enabled,
            **kwargs,
        )

        self.dimension = dimension
        self._collection = None
        self._initialized = False
        self._metadata_file: Path = self.db_path / f"{store_name}_file_metadata.json"
        self._metadata_cache: dict[str, dict[str, FileMetadata]] = {}

    @property
    def collection_name(self) -> str:
        """Get the name of the zvec collection for this store."""
        return f"chunks_{self.store_name}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize zvec engine and open the collection."""
        if not self._initialized:
            try:
                zvec.init()
            except RuntimeError:
                pass
            self._initialized = True

        self.db_path.mkdir(parents=True, exist_ok=True)
        collection_path = str(self.db_path / self.collection_name)
        option = CollectionOption(read_only=False, enable_mmap=True)

        try:
            self._collection = zvec.open(collection_path, option)
            logger.info(f"Opened existing zvec file store collection: {collection_path}")
        except Exception:
            schema = _build_file_store_schema(self.collection_name, self.dimension)
            self._collection = zvec.create_and_open(
                path=collection_path,
                schema=schema,
                option=option,
            )
            logger.info(f"Created new zvec file store collection: {collection_path}")

        self._metadata_cache = await self._load_metadata()

    async def close(self) -> None:
        """Close zvec collection and persist metadata."""
        if self._metadata_cache:
            await self._save_metadata(self._metadata_cache)

        if self._collection is not None:
            try:
                self._collection.flush()
            except Exception as e:
                logger.warning(f"Failed to flush collection on close: {e}")
            self._collection = None

    # ------------------------------------------------------------------
    # Metadata management
    # ------------------------------------------------------------------

    async def _load_metadata(self) -> dict[str, dict[str, FileMetadata]]:
        """Load file metadata from JSON file."""
        if not self._metadata_file.exists():
            return {}
        try:
            data = json.loads(self._metadata_file.read_text(encoding="utf-8"))
            result: dict[str, dict[str, FileMetadata]] = {}
            for source, files in data.items():
                result[source] = {}
                for path, meta in files.items():
                    result[source][path] = FileMetadata(**meta)
            return result
        except Exception as e:
            logger.warning(f"Failed to load metadata from {self._metadata_file}: {e}")
            return {}

    async def _save_metadata(self, metadata: dict[str, dict[str, FileMetadata]]) -> None:
        """Save file metadata to JSON file."""
        try:
            out: dict[str, dict[str, dict]] = {}
            for source, files in metadata.items():
                out[source] = {}
                for path, meta in files.items():
                    out[source][path] = {
                        "path": meta.path,
                        "hash": meta.hash,
                        "mtime_ms": meta.mtime_ms,
                        "size": meta.size,
                        "chunk_count": meta.chunk_count,
                    }
            self._metadata_file.write_text(
                json.dumps(out, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save metadata to {self._metadata_file}: {e}")

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def upsert_file(
        self,
        file_meta: FileMetadata,
        source: MemorySource,
        chunks: list[MemoryChunk],
    ) -> None:
        """Insert or update a file and its chunks."""
        if not chunks:
            return

        # Delete existing chunks for this file first
        await self.delete_file(file_meta.path, source)

        # Generate embeddings
        chunks = await self.get_chunk_embeddings(chunks)

        file_meta_json = json.dumps(
            {
                "path": file_meta.path,
                "hash": file_meta.hash,
                "mtime_ms": file_meta.mtime_ms,
                "size": file_meta.size,
                "chunk_count": len(chunks),
            },
            ensure_ascii=False,
        )

        docs = [_chunk_to_doc(c, file_meta_json) for c in chunks]
        self._collection.insert(docs)

        # Update metadata cache
        if source.value not in self._metadata_cache:
            self._metadata_cache[source.value] = {}
        self._metadata_cache[source.value][file_meta.path] = FileMetadata(
            hash=file_meta.hash,
            mtime_ms=file_meta.mtime_ms,
            size=file_meta.size,
            path=file_meta.path,
            chunk_count=len(chunks),
        )

    async def delete_file(self, path: str, source: MemorySource) -> None:
        """Delete a file and all its chunks."""
        filter_expr = f"path='{_escape(path)}' AND source='{_escape(source.value)}'"
        results = self._collection.query(topk=_ZVEC_MAX_TOPK, filter=filter_expr, include_vector=False)

        ids_to_delete = [doc.id for doc in results]
        if ids_to_delete:
            self._collection.delete(ids_to_delete)

        if source.value in self._metadata_cache:
            self._metadata_cache[source.value].pop(path, None)

    async def delete_file_chunks(self, path: str, chunk_ids: list[str]) -> None:
        """Delete specific chunks for a file."""
        if not chunk_ids:
            return
        self._collection.delete(chunk_ids)

    async def upsert_chunks(
        self,
        chunks: list[MemoryChunk],
        source: MemorySource,
    ) -> None:
        """Insert or update specific chunks."""
        if not chunks:
            return

        chunks = await self.get_chunk_embeddings(chunks)
        docs = [_chunk_to_doc(c) for c in chunks]
        self._collection.upsert(docs)

    # ------------------------------------------------------------------
    # Listing and metadata
    # ------------------------------------------------------------------

    async def list_files(self, source: MemorySource) -> list[str]:
        """List all indexed files for a source."""
        if source.value not in self._metadata_cache:
            return []
        return list(self._metadata_cache[source.value].keys())

    async def get_file_metadata(
        self,
        path: str,
        source: MemorySource,
    ) -> FileMetadata | None:
        """Get file metadata."""
        if source.value not in self._metadata_cache:
            return None
        return self._metadata_cache[source.value].get(path)

    async def update_file_metadata(self, file_meta: FileMetadata, source: MemorySource) -> None:
        """Update file metadata without affecting chunks."""
        if source.value not in self._metadata_cache:
            self._metadata_cache[source.value] = {}
        self._metadata_cache[source.value][file_meta.path] = FileMetadata(
            hash=file_meta.hash,
            mtime_ms=file_meta.mtime_ms,
            size=file_meta.size,
            path=file_meta.path,
            chunk_count=file_meta.chunk_count,
        )

    async def get_file_chunks(
        self,
        path: str,
        source: MemorySource,
    ) -> list[MemoryChunk]:
        """Get all chunks for a file."""
        filter_expr = f"path='{_escape(path)}' AND source='{_escape(source.value)}'"
        results = self._collection.query(
            topk=_ZVEC_MAX_TOPK,
            filter=filter_expr,
            include_vector=True,
        )
        chunks = [_doc_to_chunk(doc) for doc in results]
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform vector similarity search."""
        if not self.vector_enabled or not query:
            return []

        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []

        filter_expr = _build_source_filter(sources)
        vq = VectorQuery(field_name=_DEFAULT_VECTOR_FIELD, vector=query_embedding)

        try:
            results = self._collection.query(
                vectors=vq,
                topk=min(limit, _ZVEC_MAX_TOPK),
                filter=filter_expr,
                include_vector=False,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        search_results = []
        for doc in results:
            score = doc.score if doc.score is not None else 0.0
            # zvec cosine score might need normalization depending on version
            search_results.append(
                MemorySearchResult(
                    path=str(doc.field("path") or ""),
                    start_line=int(doc.field("start_line") or 0),
                    end_line=int(doc.field("end_line") or 0),
                    score=score,
                    snippet=str(doc.field("content") or ""),
                    source=MemorySource(str(doc.field("source") or "")),
                    raw_metric=score,
                ),
            )

        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results[:limit]

    async def keyword_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform keyword search via Python substring matching.

        Fetches chunks from zvec (subject to topk limit) then matches
        keywords in Python. For collections larger than the topk limit,
        not all documents are scanned.
        """
        if not self.fts_enabled or not query:
            return []

        words = query.split()
        if not words:
            return []

        # Fetch candidate chunks from zvec
        filter_expr = _build_source_filter(sources)
        results = self._collection.query(
            topk=_ZVEC_MAX_TOPK,
            filter=filter_expr,
            include_vector=False,
        )

        query_lower = query.lower()
        words_lower = [w.lower() for w in words]
        n_words = len(words)

        search_results = []
        for doc in results:
            text = str(doc.field("content") or "")
            text_lower = text.lower()
            match_count = sum(1 for w in words_lower if w in text_lower)
            if match_count == 0:
                continue

            base_score = match_count / n_words
            phrase_bonus = 0.2 if n_words > 1 and query_lower in text_lower else 0.0
            score = min(1.0, base_score + phrase_bonus)

            search_results.append(
                MemorySearchResult(
                    path=str(doc.field("path") or ""),
                    start_line=int(doc.field("start_line") or 0),
                    end_line=int(doc.field("end_line") or 0),
                    score=score,
                    snippet=text,
                    source=MemorySource(str(doc.field("source") or "")),
                ),
            )

        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results[:limit]

    async def hybrid_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
        vector_weight: float = 0.7,
        candidate_multiplier: float = 3.0,
    ) -> list[MemorySearchResult]:
        """Perform hybrid search combining vector and keyword search."""
        assert 0.0 <= vector_weight <= 1.0, f"vector_weight must be between 0 and 1, got {vector_weight}"

        candidates = min(200, max(1, int(limit * candidate_multiplier)))
        text_weight = 1.0 - vector_weight

        if self.vector_enabled and self.fts_enabled:
            keyword_results = await self.keyword_search(query, candidates, sources)
            vector_results = await self.vector_search(query, candidates, sources)

            if not keyword_results:
                return vector_results[:limit]
            elif not vector_results:
                return keyword_results[:limit]
            else:
                return self._merge_hybrid_results(
                    vector=vector_results,
                    keyword=keyword_results,
                    vector_weight=vector_weight,
                    text_weight=text_weight,
                )[:limit]
        elif self.vector_enabled:
            return await self.vector_search(query, limit, sources)
        elif self.fts_enabled:
            return await self.keyword_search(query, limit, sources)
        else:
            return []

    @staticmethod
    def _merge_hybrid_results(
        vector: list[MemorySearchResult],
        keyword: list[MemorySearchResult],
        vector_weight: float,
        text_weight: float,
    ) -> list[MemorySearchResult]:
        """Merge vector and keyword search results with weighted scoring."""
        merged: dict[str, MemorySearchResult] = {}

        for result in vector:
            result.score = result.score * vector_weight
            merged[result.merge_key] = result

        for result in keyword:
            key = result.merge_key
            if key in merged:
                merged[key].score += result.score * text_weight
            else:
                result.score = result.score * text_weight
                merged[key] = result

        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def clear_all(self) -> None:
        """Clear all indexed data."""
        # Delete all documents
        stats = self._collection.stats
        count = stats.doc_count if stats else 0
        if count > 0:
            try:
                self._collection.delete_by_filter("content!=''")
            except Exception:
                remaining = count
                while remaining > 0:
                    batch = self._collection.query(
                        topk=min(remaining, _ZVEC_MAX_TOPK),
                        include_vector=False,
                    )
                    if not batch:
                        break
                    self._collection.delete([doc.id for doc in batch])
                    remaining -= len(batch)

        self._metadata_cache = {}
        await self._save_metadata({})
        logger.info(f"Cleared all data from zvec file store: {self.collection_name}")
