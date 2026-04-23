"""Pure-Python file storage with JSONL persistence."""

import json
from pathlib import Path

import numpy as np

from .base_file_store import BaseFileStore
from ..component_registry import R
from ...schema import FileChunk, FileMetadata, SearchFilter
from ...utils import batch_cosine_similarity


@R.register("local")
class LocalFileStore(BaseFileStore):
    """In-memory file storage with JSONL disk persistence.

    No external database required. All data lives in Python dicts;
    writes are flushed to JSONL files on disk and survive restarts.
    """

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self._encoding: str = encoding
        self._chunks: dict[str, FileChunk] = {}
        self._files: dict[str, FileMetadata] = {}
        self._chunks_file: Path = self.db_path / f"{self.store_name}_chunks.jsonl"
        self._metadata_file: Path = self.db_path / f"{self.store_name}_file_metadata.json"

    # -- Persistence helpers ------------------------------------------------

    async def _load_chunks(self) -> None:
        """Load chunks from JSONL file into memory."""
        if not self._chunks_file.exists():
            return
        try:
            data = self._chunks_file.read_text(encoding=self._encoding)
            self._chunks = {}
            for line in data.strip().split("\n"):
                if not line:
                    continue
                chunk = FileChunk.model_validate(json.loads(line))
                self._chunks[chunk.id] = chunk
        except Exception as e:
            self.logger.warning(f"Failed to load chunks: {e}")

    async def _save_chunks(self) -> None:
        """Persist chunks to JSONL file with atomic write."""
        lines = [json.dumps(c.model_dump(mode="json"), ensure_ascii=False) for c in self._chunks.values()]
        content = "\n".join(lines)
        temp_path = self._chunks_file.with_suffix(".tmp")
        try:
            temp_path.write_text(content, encoding=self._encoding)
            temp_path.replace(self._chunks_file)
        except Exception as e:
            self.logger.error(f"Failed to save chunks: {e}")
            raise
        finally:
            if temp_path.exists():
                temp_path.unlink()

    async def _load_metadata(self) -> None:
        """Load file metadata from JSON file into memory."""
        if not self._metadata_file.exists():
            return
        try:
            data = self._metadata_file.read_text(encoding=self._encoding)
            raw: dict = json.loads(data)
            self._files = {path: FileMetadata(**meta) for path, meta in raw.items()}
        except Exception as e:
            self.logger.warning(f"Failed to load metadata: {e}")

    async def _save_metadata(self) -> None:
        """Persist file metadata to JSON file with atomic write."""
        raw = {path: meta.model_dump(exclude={"content"}, mode="json") for path, meta in self._files.items()}
        content = json.dumps(raw, indent=2, ensure_ascii=False)
        temp_path = self._metadata_file.with_suffix(".tmp")
        try:
            temp_path.write_text(content, encoding=self._encoding)
            temp_path.replace(self._metadata_file)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            raise
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # -- Lifecycle ----------------------------------------------------------

    async def _start(self) -> None:
        """Load persisted data into memory."""
        await self._load_metadata()
        await self._load_chunks()
        self.logger.info(
            f"LocalFileStore '{self.store_name}' ready: "
            f"{len(self._chunks)} chunks, metadata at {self._metadata_file}",
        )
        await super()._start()

    async def _close(self) -> None:
        """Flush state to disk and clear memory."""
        await self._save_metadata()
        await self._save_chunks()
        self._chunks.clear()
        self._files.clear()
        await super()._close()

    # -- Write operations ---------------------------------------------------

    async def upsert_file(self, file_meta: FileMetadata, chunks: list[FileChunk]) -> None:
        """Insert or update a file and its chunks."""
        await self.delete_file(file_meta.path)

        if chunks:
            chunks = await self.get_chunk_embeddings(chunks)
            for chunk in chunks:
                self._chunks[chunk.id] = chunk

        file_meta.chunk_count = len(chunks)
        if file_meta.path:
            self._files[file_meta.path] = FileMetadata(
                hash=file_meta.hash,
                mtime_ms=file_meta.mtime_ms,
                size=file_meta.size,
                path=file_meta.path,
                chunk_count=file_meta.chunk_count,
                metadata=file_meta.metadata,
            )

    async def delete_file(self, path: str) -> None:
        """Delete a file and all its chunks."""
        to_delete = [cid for cid, chunk in self._chunks.items() if chunk.path == path]
        for cid in to_delete:
            del self._chunks[cid]
        self._files.pop(path, None)

    # -- Read operations ----------------------------------------------------

    async def list_files(self) -> list[str]:
        """List all indexed file paths."""
        return list(self._files.keys())

    async def get_file_metadata(self, path: str) -> FileMetadata | None:
        """Get metadata for a specific file."""
        return self._files.get(path)

    async def get_file_chunks(self, path: str) -> list[FileChunk]:
        """Get all chunks for a file, sorted by start_line."""
        chunks = [chunk for chunk in self._chunks.values() if chunk.path == path]
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    # -- Search operations --------------------------------------------------

    async def vector_search(self, query: str, limit: int, search_filter: SearchFilter | None = None) -> list[FileChunk]:
        """Cosine-similarity vector search over in-memory embeddings."""
        if not self.vector_enabled or not query:
            return []

        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return []

        candidates = self._apply_filter(
            [c for c in self._chunks.values() if c.embedding],
            search_filter,
        )
        if not candidates:
            return []

        expected_dim = self.embedding_dim

        # Validate and align embedding dimensions
        valid_embeddings = []
        for chunk in candidates:
            emb = chunk.embedding
            emb_len = len(emb)
            if emb_len != expected_dim:
                emb = (emb + [0.0] * (expected_dim - emb_len)) if emb_len < expected_dim else emb[:expected_dim]
            valid_embeddings.append(emb)

        query_array = np.array([query_embedding])
        chunk_embeddings = np.array(valid_embeddings)
        similarities = batch_cosine_similarity(query_array, chunk_embeddings)[0]

        results = []
        for chunk, sim in zip(candidates, similarities):
            results.append(
                FileChunk(
                    id=chunk.id,
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    hash=chunk.hash,
                    text=chunk.text,
                    embedding=chunk.embedding,
                    scores={"vector": float(sim), "score": float(sim)},
                ),
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def keyword_search(
        self,
        query: str,
        limit: int,
        search_filter: SearchFilter | None = None,
    ) -> list[FileChunk]:
        """Keyword search via substring matching."""
        if not self.fts_enabled or not query:
            return []

        words = query.split()
        if not words:
            return []

        filtered_chunks = self._apply_filter(list(self._chunks.values()), search_filter)

        results = []
        for chunk in filtered_chunks:
            score = self._score_keyword_match(query, chunk.text)
            if score == 0.0:
                continue
            results.append(
                FileChunk(
                    id=chunk.id,
                    path=chunk.path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    hash=chunk.hash,
                    text=chunk.text,
                    scores={"keyword": score, "score": score},
                ),
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def clear_all(self) -> None:
        """Clear all indexed data from memory and disk."""
        self._chunks.clear()
        self._files.clear()
        await self._save_chunks()
        await self._save_metadata()
        self.logger.info(f"Cleared all data from LocalFileStore '{self.store_name}'")
