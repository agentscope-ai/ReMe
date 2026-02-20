"""ChromaDB storage backend for memory index."""

import json
import time
from pathlib import Path

from loguru import logger

from .base_memory_store import BaseMemoryStore
from ..enumeration import MemorySource
from ..schema import FileMetadata, MemoryChunk, MemorySearchResult

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None
    Settings = None


class ChromaMemoryStore(BaseMemoryStore):
    """ChromaDB memory storage with vector and full-text search.

    Inherits embedding methods from BaseMemoryStore:
    - get_chunk_embedding / get_chunk_embeddings (async)
    - get_embedding / get_embeddings (async)

    Provides ChromaDB-backed persistent storage with:
    - Vector similarity search (native ChromaDB)
    - Full-text search (via ChromaDB where_document filter)
    - Efficient chunk and file metadata management
    """

    def __init__(
        self,
        **kwargs,
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb package is required for ChromaMemoryStore. Install it with: pip install chromadb",
            )

        super().__init__(**kwargs)
        self.client: "chromadb.ClientAPI | None" = None
        self.chunks_collection: "chromadb.Collection | None" = None
        # Initialize metadata file path (db_path and store_name are set by base class)
        self._metadata_file: Path = self.db_path.parent / f"{self.store_name}_file_metadata.json"
        self._metadata_cache: dict[str, dict[str, FileMetadata]] = {}

    @property
    def collection_name(self) -> str:
        """Get the name of the ChromaDB collection for this store."""
        return f"chunks_{self.store_name}"

    async def _load_metadata(self) -> dict[str, dict[str, FileMetadata]]:
        """Load file metadata from disk.

        Returns:
            Dictionary mapping source -> path -> FileMetadata
        """
        if not self._metadata_file.exists():
            return {}

        try:
            data = self._metadata_file.read_text(encoding="utf-8")
            metadata_dict = json.loads(data)

            # Convert dict to FileMetadata objects
            result = {}
            for source, files in metadata_dict.items():
                result[source] = {}
                for path, meta in files.items():
                    result[source][path] = FileMetadata(**meta)

            logger.debug(f"Loaded file metadata from {self._metadata_file}")
            return result
        except Exception as e:
            logger.warning(f"Failed to load file metadata from {self._metadata_file}: {e}")
            return {}

    async def _save_metadata(self, metadata: dict[str, dict[str, FileMetadata]]) -> None:
        """Save file metadata to disk.

        Args:
            metadata: Dictionary mapping source -> path -> FileMetadata
        """
        try:
            # Convert FileMetadata objects to dict for JSON serialization
            metadata_dict = {}
            for source, files in metadata.items():
                metadata_dict[source] = {}
                for path, meta in files.items():
                    metadata_dict[source][path] = {
                        "path": meta.path,
                        "hash": meta.hash,
                        "mtime_ms": meta.mtime_ms,
                        "size": meta.size,
                        "chunk_count": meta.chunk_count,
                    }

            data = json.dumps(metadata_dict, indent=2, ensure_ascii=False)
            self._metadata_file.write_text(data, encoding="utf-8")
            logger.debug(f"Saved file metadata to {self._metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save file metadata to {self._metadata_file}: {e}")

    async def start(self) -> None:
        """Initialize ChromaDB client and collection."""
        if self.client is not None:
            return

        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Get or create the chunks collection
        # ChromaDB uses cosine distance by default for similarity
        self.chunks_collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Load metadata into cache
        self._metadata_cache = await self._load_metadata()

        logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
        logger.info(f"File metadata will be persisted to: {self._metadata_file}")

    async def upsert_file(
        self,
        file_meta: FileMetadata,
        source: MemorySource,
        chunks: list[MemoryChunk],
    ) -> None:
        """Insert or update file and its chunks."""
        if not chunks:
            return

        # Delete existing chunks for this file first
        await self.delete_file(file_meta.path, source)

        # Batch generate embeddings for all chunks
        # (base class returns mock embeddings when vector_enabled=False)
        chunks = await self.get_chunk_embeddings(chunks)

        # Prepare data for ChromaDB batch upsert
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        now = int(time.time() * 1000)
        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.text)
            embeddings.append(chunk.embedding)
            metadatas.append(
                {
                    "path": file_meta.path,
                    "source": source.value,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "hash": chunk.hash,
                    "updated_at": now,
                },
            )

        # Batch upsert to ChromaDB (always pass embeddings to prevent default embedding function)
        self.chunks_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Update file metadata in cache
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
        """Delete file and all its chunks."""
        # Query for all chunks with this path and source
        results = self.chunks_collection.get(
            where={"$and": [{"path": path}, {"source": source.value}]},
            include=[],
        )

        if results["ids"]:
            self.chunks_collection.delete(
                ids=results["ids"],
            )

        # Remove from file metadata cache
        if source.value in self._metadata_cache:
            self._metadata_cache[source.value].pop(path, None)

    async def delete_file_chunks(self, path: str, chunk_ids: list[str]) -> None:
        """Delete specific chunks for a file."""
        if not chunk_ids:
            return

        self.chunks_collection.delete(
            ids=chunk_ids,
        )

        # Update chunk count in file metadata cache
        for source_meta in self._metadata_cache.values():
            if path in source_meta:
                # Recalculate chunk count
                results = self.chunks_collection.get(
                    where={"path": path},
                    include=[],
                )
                source_meta[path].chunk_count = len(results["ids"])
                break

    async def upsert_chunks(
        self,
        chunks: list[MemoryChunk],
        source: MemorySource,
    ) -> None:
        """Insert or update specific chunks without affecting other chunks."""
        if not chunks:
            return

        # Batch generate embeddings for all chunks
        # (base class returns mock embeddings when vector_enabled=False)
        chunks = await self.get_chunk_embeddings(chunks)

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        now = int(time.time() * 1000)
        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.text)
            embeddings.append(chunk.embedding)
            metadatas.append(
                {
                    "path": chunk.path,
                    "source": source.value,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "hash": chunk.hash,
                    "updated_at": now,
                },
            )

        # Always pass embeddings to prevent default embedding function
        self.chunks_collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

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
        """Get file metadata with chunk count."""
        if source.value not in self._metadata_cache:
            return None
        return self._metadata_cache[source.value].get(path)

    async def get_file_chunks(
        self,
        path: str,
        source: MemorySource,
    ) -> list[MemoryChunk]:
        """Get all chunks for a file."""
        results = self.chunks_collection.get(
            where={"$and": [{"path": path}, {"source": source.value}]},
            include=["documents", "embeddings", "metadatas"],
        )

        chunks = []
        for i, chunk_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            chunks.append(
                MemoryChunk(
                    id=chunk_id,
                    path=metadata["path"],
                    source=MemorySource(metadata["source"]),
                    start_line=metadata["start_line"],
                    end_line=metadata["end_line"],
                    text=results["documents"][i],
                    hash=metadata["hash"],
                    embedding=results["embeddings"][i] if results["embeddings"] is not None else None,
                ),
            )

        # Sort by start_line
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    async def vector_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform vector similarity search."""
        if not self.vector_enabled or not query:
            return []

        # Get query embedding
        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []

        # Build where filter for sources
        where_filter = None
        if sources:
            if len(sources) == 1:
                where_filter = {"source": sources[0].value}
            else:
                where_filter = {"source": {"$in": [s.value for s in sources]}}

        # Perform vector search
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, _ in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                # Convert cosine distance to similarity score
                # Cosine distance range is [0, 2], convert to [1, 0] score
                score = max(0.0, 1.0 - distance / 2.0)

                search_results.append(
                    MemorySearchResult(
                        path=metadata["path"],
                        start_line=metadata["start_line"],
                        end_line=metadata["end_line"],
                        score=score,
                        snippet=results["documents"][0][i],
                        source=MemorySource(metadata["source"]),
                        raw_metric=distance,
                    ),
                )

        # Sort by score descending
        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    async def keyword_search(
        self,
        query: str,
        limit: int,
        sources: list[MemorySource] | None = None,
    ) -> list[MemorySearchResult]:
        """Perform keyword/full-text search.

        ChromaDB supports where_document filter for text matching.
        Note: ChromaDB's $contains is case-sensitive, so we generate multiple
        case variants (original, lowercase, capitalized) for each word to
        improve recall while maintaining case-insensitive scoring.
        """
        if not self.fts_enabled or not query:
            return []

        # Normalize whitespace and split into words
        words = query.split()
        if not words:
            return []

        # Generate case variants for each word to handle case-sensitive $contains
        # Include: original, lowercase, and capitalized forms
        word_variants = set()
        for word in words:
            word_variants.add(word)  # original
            word_variants.add(word.lower())  # lowercase
            word_variants.add(word.capitalize())  # Capitalized
            word_variants.add(word.upper())  # UPPERCASE
        word_variants_list = list(word_variants)

        # Build where filter for sources
        where_filter = None
        if sources:
            if len(sources) == 1:
                where_filter = {"source": sources[0].value}
            else:
                where_filter = {"source": {"$in": [s.value for s in sources]}}

        # ChromaDB where_document uses $contains for substring matching (case-sensitive)
        # Use multiple case variants to improve recall
        if len(word_variants_list) == 1:
            where_document = {"$contains": word_variants_list[0]}
        else:
            where_document = {"$or": [{"$contains": w} for w in word_variants_list]}

        # Get all matching documents
        results = self.chunks_collection.get(
            where=where_filter,
            where_document=where_document,
            include=["documents", "metadatas"],
        )

        search_results = []
        query_lower = query.lower()
        words_lower = [w.lower() for w in words]  # lowercase words for scoring
        n_words = len(words)

        for i, _ in enumerate(results["ids"]):
            metadata = results["metadatas"][i]
            text = results["documents"][i]
            text_lower = text.lower()

            # Calculate relevance score based on word matches
            match_count = sum(1 for w in words_lower if w in text_lower)
            base_score = match_count / n_words

            # Bonus for full phrase match (only applies to multi-word queries)
            phrase_bonus = 0.2 if n_words > 1 and query_lower in text_lower else 0.0
            # Scale base_score and add phrase bonus, max score is 1.0
            score = min(1.0, base_score + phrase_bonus)

            search_results.append(
                MemorySearchResult(
                    path=metadata["path"],
                    start_line=metadata["start_line"],
                    end_line=metadata["end_line"],
                    score=score,
                    snippet=text,
                    source=MemorySource(metadata["source"]),
                ),
            )

        # Sort by score descending and limit results
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
        """Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query text
            limit: Maximum number of results
            sources: Optional list of sources to filter
            vector_weight: Weight for vector search results (0.0-1.0).
                          Keyword weight = 1.0 - vector_weight.
            candidate_multiplier: Multiplier for candidate pool size.

        Returns:
            List of search results sorted by combined relevance score
        """
        assert 0.0 <= vector_weight <= 1.0, f"vector_weight must be between 0 and 1, got {vector_weight}"

        candidates = min(200, max(1, int(limit * candidate_multiplier)))
        text_weight = 1.0 - vector_weight

        # Perform search based on enabled backends
        if self.vector_enabled and self.fts_enabled:
            keyword_results = await self.keyword_search(query, candidates, sources)
            vector_results = await self.vector_search(query, candidates, sources)

            # Log original vector results
            logger.info("\n=== Vector Search Results ===")
            for i, r in enumerate(vector_results[:10], 1):
                snippet_preview = (r.snippet[:100] + "...") if len(r.snippet) > 100 else r.snippet
                logger.info(f"{i}. Score: {r.score:.4f} | Snippet: {snippet_preview}")

            # Log original keyword results
            logger.info("\n=== Keyword Search Results ===")
            for i, r in enumerate(keyword_results[:10], 1):
                snippet_preview = (r.snippet[:100] + "...") if len(r.snippet) > 100 else r.snippet
                logger.info(f"{i}. Score: {r.score:.4f} | Snippet: {snippet_preview}")

            if not keyword_results:
                return vector_results[:limit]
            elif not vector_results:
                return keyword_results[:limit]
            else:
                merged = self._merge_hybrid_results(
                    vector=vector_results,
                    keyword=keyword_results,
                    vector_weight=vector_weight,
                    text_weight=text_weight,
                )

                # Log merged results
                logger.info("\n=== Merged Hybrid Results ===")
                for i, r in enumerate(merged[:10], 1):
                    snippet_preview = (r.snippet[:100] + "...") if len(r.snippet) > 100 else r.snippet
                    logger.info(f"{i}. Score: {r.score:.4f} | Snippet: {snippet_preview}")

                return merged[:limit]
        elif self.vector_enabled:
            vector_results = await self.vector_search(query, limit, sources)
            return vector_results
        elif self.fts_enabled:
            keyword_results = await self.keyword_search(query, limit, sources)
            return keyword_results
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

        # Process vector results
        for result in vector:
            result.score = result.score * vector_weight
            merged[result.merge_key] = result

        # Process keyword results
        for result in keyword:
            key = result.merge_key
            if key in merged:
                merged[key].score += result.score * text_weight
            else:
                result.score = result.score * text_weight
                merged[key] = result

        # Sort by score and return
        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def clear_all(self) -> None:
        """Clear all indexed data."""
        # Delete and recreate the collection
        self.client.delete_collection(
            name=self.collection_name,
        )
        self.chunks_collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Clear file metadata cache and disk
        self._metadata_cache = {}
        await self._save_metadata({})

        logger.info(f"Cleared all data from ChromaDB collection: {self.collection_name}")

    async def close(self) -> None:
        """Close ChromaDB client and release resources."""
        # Persist metadata cache to disk before closing
        if self._metadata_cache:
            await self._save_metadata(self._metadata_cache)

        # ChromaDB PersistentClient handles persistence automatically
        self.client = None
        self.chunks_collection = None
        await super().close()
