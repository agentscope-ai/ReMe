"""In-memory file store with JSONL persistence on close."""

from pathlib import Path

import aiofiles
import numpy as np
from pydantic import BaseModel

from .base_file_store import BaseFileStore
from ..component_registry import R
from ...schema import FileChunk, FileNode
from ...utils import batch_cosine_similarity, get_logger

logger = get_logger()


@R.register("local")
class LocalFileStore(BaseFileStore):
    """In-memory file store with deferred JSONL persistence."""

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self._encoding = encoding
        self._nodes: dict[str, FileNode] = {}
        self._chunks: dict[str, FileChunk] = {}
        self._nodes_file = self.store_path / "nodes.jsonl"
        self._chunks_file = self.store_path / "chunks.jsonl"

    # Lifecycle

    async def _start(self) -> None:
        await super()._start()
        await self._load(self._nodes_file, self._nodes, FileNode, "path")
        await self._load(self._chunks_file, self._chunks, FileChunk, "id")
        self.logger.info(f"LocalFileStore '{self.store_name}' ready: "
                         f"{len(self._nodes)} nodes, {len(self._chunks)} chunks")

    async def _close(self) -> None:
        await self._dump(self._nodes_file, list(self._nodes.values()))
        await self._dump(self._chunks_file, list(self._chunks.values()))
        self._nodes.clear()
        self._chunks.clear()
        await super()._close()

    async def _load(self, file: Path, target: dict, model: type[BaseModel], key: str) -> None:
        if not file.exists():
            return

        target.clear()
        try:
            async with aiofiles.open(file, encoding=self._encoding) as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        obj = model.model_validate_json(line)
                        target[getattr(obj, key)] = obj
        except Exception as e:
            self.logger.exception(f"Failed to load {file}: {e}")

    async def _dump(self, file: Path, items: list[BaseModel]) -> None:
        try:
            content = "\n".join(o.model_dump_json() for o in items)
            tmp = file.with_suffix(".tmp")
            async with aiofiles.open(tmp, "w", encoding=self._encoding) as f:
                await f.write(content)
            tmp.replace(file)
        except Exception as e:
            self.logger.exception(f"Failed to write {file}: {e}")

    # Node operations

    async def upsert_node(self, node: FileNode) -> None:
        self._nodes[node.path] = node

    async def get_node_by_path(self, path: str) -> FileNode | None:
        return self._nodes.get(path)

    async def delete_node_by_path(self, path: str) -> FileNode | None:
        return self._nodes.pop(path, None)

    # Chunk operations

    async def upsert_chunks_with_path(self, path: str, chunks: list[FileChunk]) -> None:
        existing = await self.get_chunks_by_path(path)
        cached = {c.id: c.embedding for c in existing if c.embedding}

        await self.delete_chunks_by_path(path)
        if not chunks:
            return

        needs_embed: list[FileChunk] = []
        for c in chunks:
            if c.embedding:
                continue
            if c.id in cached:
                c.embedding = cached[c.id]
            elif c.text:
                needs_embed.append(c)

        if needs_embed:
            embeddings = await self.get_embeddings([c.text for c in needs_embed])
            if embeddings:
                for c, emb in zip(needs_embed, embeddings):
                    if emb:
                        c.embedding = emb

        for c in chunks:
            self._chunks[c.id] = c

    async def get_chunks_by_path(self, path: str) -> list[FileChunk]:
        chunks = [c for c in self._chunks.values() if c.path == path]
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    async def delete_chunks_by_path(self, path: str) -> None:
        stale = [cid for cid, c in self._chunks.items() if c.path == path]
        for cid in stale:
            del self._chunks[cid]

    # Search

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if self.embedding_model is None or not query:
            return []

        query_embedding = await self.embedding_model.get_embedding(query)
        if not query_embedding:
            return []

        candidates = [c for c in self._chunks.values() if c.embedding]
        emb_missing = [c for c in self._chunks.values() if not c.embedding]
        if emb_missing:
            logger.warning(f"Embedding missing for {len(emb_missing)} chunks")
        if not candidates:
            return []

        chunk_embs = np.array([c.embedding for c in candidates])
        similarities = batch_cosine_similarity(np.array([query_embedding]), chunk_embs)[0]

        results = [
            c.model_copy(update={"scores": {"vector": float(s), "score": float(s)}})
            for c, s in zip(candidates, similarities)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if not self.fts_enabled or self.bm25 is None:
            return []

        query = query.strip()
        if not query:
            return []

        doc_id_score_dict = self.bm25.retrieve(query, limit=limit)
        results = []
        for doc_id, score in doc_id_score_dict.items():
            chunk = self._chunks.get(doc_id)
            if chunk:
                results.append(chunk.model_copy(update={"scores": {"keyword": score, "score": score}}))

        return results

    # Reindex

    async def reindex_vector(self) -> None:
        if not self.vector_enabled or self.embedding_model is None:
            return

        await self.embedding_model.get_node_embeddings(list(self._chunks.values()))

    async def reindex_keyword(self) -> None:
        if not self.fts_enabled or self.bm25 is None:
            return

