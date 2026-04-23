"""ChromaDB storage backend for file store."""

import json
import time
from pathlib import Path

from .base_file_store import BaseFileStore
from ..component_registry import R
from ...schema import FileChunk, FileMetadata, SearchFilter

try:
    import chromadb
    from chromadb.config import Settings

    _CHROMADB_IMPORT_ERROR: Exception | None = None
except Exception as e:
    _CHROMADB_IMPORT_ERROR = e
    chromadb = None
    Settings = None


@R.register("chroma")
class ChromaFileStore(BaseFileStore):
    """ChromaDB file storage with vector and full-text search.

    Uses ChromaDB's native vector search and `where_document` $contains
    for keyword matching. File metadata is persisted to a JSON file
    alongside the ChromaDB database.
    """

    def __init__(self, **kwargs):
        if _CHROMADB_IMPORT_ERROR is not None:
            raise _CHROMADB_IMPORT_ERROR
        super().__init__(**kwargs)
        self.client: "chromadb.ClientAPI | None" = None
        self.chunks_collection: "chromadb.Collection | None" = None
        self._metadata_file: Path = self.db_path / f"{self.store_name}_file_metadata.json"
        self._metadata_cache: dict[str, FileMetadata] = {}

    @property
    def collection_name(self) -> str:
        return f"chunks_{self.store_name}"

    # -- Persistence helpers ------------------------------------------------

    async def _load_metadata(self) -> None:
        if not self._metadata_file.exists():
            return
        try:
            data = self._metadata_file.read_text(encoding="utf-8")
            raw = json.loads(data)
            self._metadata_cache = {path: FileMetadata(**meta) for path, meta in raw.items()}
        except Exception as e:
            self.logger.warning(f"Failed to load metadata: {e}")

    async def _save_metadata(self) -> None:
        try:
            raw = {
                path: meta.model_dump(exclude={"content"}, mode="json")
                for path, meta in self._metadata_cache.items()
            }
            data = json.dumps(raw, indent=2, ensure_ascii=False)
            temp = self._metadata_file.with_suffix(".tmp")
            temp.write_text(data, encoding="utf-8")
            temp.replace(self._metadata_file)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            raise
        finally:
            if temp.exists():
                temp.unlink()

    # -- Lifecycle ----------------------------------------------------------

    async def _start(self, app_context=None) -> None:
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )
        self.chunks_collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        await self._load_metadata()
        self.logger.info(
            f"ChromaFileStore '{self.store_name}' ready: "
            f"collection={self.collection_name}, metadata at {self._metadata_file}",
        )
        await super()._start(app_context)

    async def _close(self) -> None:
        await self._save_metadata()
        self.client = None
        self.chunks_collection = None
        await super()._close()

    # -- Write operations ---------------------------------------------------

    async def upsert_file(self, file_meta: FileMetadata, chunks: list[FileChunk]) -> None:
        # Always delete existing data for this file first
        if file_meta.path:
            await self.delete_file(file_meta.path)

        if chunks:
            chunks = await self.get_chunk_embeddings(chunks)

            ids, documents, embeddings, metadatas = [], [], [], []
            now = int(time.time() * 1000)
            for chunk in chunks:
                ids.append(chunk.id)
                documents.append(chunk.text)
                embeddings.append(chunk.embedding if chunk.embedding else [0.0] * self.embedding_dim)
                metadatas.append({
                    "path": file_meta.path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "hash": chunk.hash,
                    "updated_at": now,
                })

            self.chunks_collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        file_meta.chunk_count = len(chunks)
        if file_meta.path:
            self._metadata_cache[file_meta.path] = FileMetadata(
                hash=file_meta.hash,
                modified_time=file_meta.modified_time,
                size=file_meta.size,
                path=file_meta.path,
                chunk_count=file_meta.chunk_count,
                metadata=file_meta.metadata,
            )

    async def delete_file(self, path: str) -> None:
        results = self.chunks_collection.get(where={"path": path}, include=[])
        if results["ids"]:
            self.chunks_collection.delete(ids=results["ids"])
        self._metadata_cache.pop(path, None)

    # -- Read operations ----------------------------------------------------

    async def list_files(self) -> list[str]:
        """List all indexed file paths."""
        return list(self._metadata_cache.keys())

    async def get_file_metadata(self, path: str) -> FileMetadata | None:
        """Get metadata for a specific file."""
        return self._metadata_cache.get(path)

    # -- Search helpers -----------------------------------------------------

    def _chunk_from_chroma(self, chunk_id: str, md: dict, text: str, embedding=None) -> FileChunk:
        return FileChunk(
            id=chunk_id,
            path=md["path"],
            start_line=md["start_line"],
            end_line=md["end_line"],
            text=text,
            hash=md["hash"],
            embedding=embedding,
        )

    # -- Search operations --------------------------------------------------

    async def vector_search(self, query: str, limit: int, search_filter: SearchFilter | None = None) -> list[FileChunk]:
        if not self.vector_enabled or not query:
            return []

        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []

        try:
            results = self.chunks_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                md = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                score = max(0.0, 1.0 - distance / 2.0)

                chunk = self._chunk_from_chroma(cid, md, results["documents"][0][i])
                chunk.scores = {"vector": score, "score": score}
                chunks.append(chunk)

        chunks = self._apply_filter(chunks, search_filter)
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:limit]

    async def keyword_search(
            self,
            query: str,
            limit: int,
            search_filter: SearchFilter | None = None,
    ) -> list[FileChunk]:
        """Keyword search via ChromaDB $contains with case variants."""
        if not self.fts_enabled or not query:
            return []

        words = query.split()
        if not words:
            return []

        # Generate case variants for case-insensitive matching
        word_variants = set()
        for word in words:
            word_variants.add(word)
            word_variants.add(word.lower())
            word_variants.add(word.capitalize())
            word_variants.add(word.upper())
        variants_list = list(word_variants)

        if len(variants_list) == 1:
            where_document: dict = {"$contains": variants_list[0]}
        else:
            where_document = {"$or": [{"$contains": w} for w in variants_list]}

        results = self.chunks_collection.get(
            where_document=where_document,
            include=["documents", "metadatas"],
        )

        chunks = []
        for i, cid in enumerate(results["ids"]):
            md = results["metadatas"][i]
            text = results["documents"][i]
            score = self._score_keyword_match(query, text)
            if score == 0.0:
                continue

            chunk = self._chunk_from_chroma(cid, md, text)
            chunk.scores = {"keyword": score, "score": score}
            chunks.append(chunk)

        chunks = self._apply_filter(chunks, search_filter)
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:limit]

    # -- Clear --------------------------------------------------------------

    async def clear_all(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.chunks_collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._metadata_cache = {}
        await self._save_metadata()
        self.logger.info(f"Cleared all data from ChromaFileStore '{self.store_name}'")
