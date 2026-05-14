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
        self.encoding = encoding
        self.file_chunks: dict[str, FileChunk] = {}
        self.nodes_path = self.store_path / "file_nodes.jsonl"
        self.chunks_path = self.store_path / "file_chunks.jsonl"

    # Lifecycle

    async def _start(self) -> None:
        await super()._start()
        await self._load_jsonl(self.chunks_path, self.file_chunks, FileChunk, "id")
        self.logger.info(f"LocalFileStore '{self.store_name}' ready: "
                         f"{len(self.file_nodes)} nodes, {len(self.file_chunks)} chunks")

    async def _close(self) -> None:
        await self._dump_jsonl(self.chunks_path, list(self.file_chunks.values()))
        self.file_chunks.clear()
        await super()._close()

    async def _load_jsonl(self, file: Path, target: dict, model: type[BaseModel], key: str) -> None:
        if not file.exists():
            return
        target.clear()
        try:
            async with aiofiles.open(file, encoding=self.encoding) as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        obj = model.model_validate_json(line)
                        target[getattr(obj, key)] = obj
        except Exception as e:
            self.logger.exception(f"Failed to load {file}: {e}")

    async def _dump_jsonl(self, file: Path, items: list[BaseModel]) -> None:
        try:
            content = "\n".join(o.model_dump_json() for o in items)
            tmp = file.with_suffix(".tmp")
            async with aiofiles.open(tmp, "w", encoding=self.encoding) as f:
                await f.write(content)
            tmp.replace(file)
        except Exception as e:
            self.logger.exception(f"Failed to write {file}: {e}")

    # Base class interface

    async def load_file_nodes(self) -> None:
        await self._load_jsonl(self.nodes_path, self.file_nodes, FileNode, "path")

    async def dump_file_nodes(self) -> None:
        await self._dump_jsonl(self.nodes_path, list(self.file_nodes.values()))

    async def upsert_file(
            self,
            file: tuple[FileNode, list[FileChunk]] | list[tuple[FileNode, list[FileChunk]]],
    ) -> None:
        if isinstance(file, tuple):
            file = [file]
        for node, chunks in file:
            old_node = self.file_nodes.pop(node.path, None)
            cached = {}
            if old_node and self.vector_enabled:
                for cid in old_node.chunk_ids:
                    old = self.file_chunks.pop(cid, None)
                    if old and old.embedding:
                        cached[cid] = old.embedding

            node.chunk_ids = []
            needs_embed = []
            for c in chunks:
                if self.vector_enabled and not c.embedding:
                    if c.id in cached:
                        c.embedding = cached[c.id]
                    elif c.text:
                        needs_embed.append(c)
                node.chunk_ids.append(c.id)
                self.file_chunks[c.id] = c
            self.file_nodes[node.path] = node

            if needs_embed and self.embedding_model:
                await self.embedding_model.get_node_embeddings(needs_embed)

            if self.fts_enabled and self.keyword_index:
                await self.keyword_index.add_docs({c.id: c.text for c in chunks if c.text})

    async def delete_by_path(self, path: str | list[str]) -> None:
        if isinstance(path, str):
            path = [path]
        deleted_chunk_ids: list[str] = []
        for p in path:
            node = self.file_nodes.pop(p, None)
            if node:
                for cid in node.chunk_ids:
                    self.file_chunks.pop(cid, None)
                    deleted_chunk_ids.append(cid)

        if self.fts_enabled and self.keyword_index and deleted_chunk_ids:
            await self.keyword_index.delete_docs(deleted_chunk_ids)

    async def clear(self) -> None:
        self.file_nodes.clear()
        self.file_chunks.clear()
        if self.fts_enabled and self.keyword_index:
            await self.keyword_index.clear()

    # Search

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if self.embedding_model is None or not query:
            return []

        query_embedding = await self.embedding_model.get_embedding(query)
        if not query_embedding:
            return []

        candidates = [c for c in self.file_chunks.values() if c.embedding is not None]
        if not candidates:
            return []

        candidate_embeddings = np.stack([c.embedding for c in candidates])
        similarities = batch_cosine_similarity(query_embedding.reshape(1, -1), candidate_embeddings)[0]

        results = [
            c.model_copy(update={"scores": {"vector": float(s), "score": float(s)}})
            for c, s in zip(candidates, similarities)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if not self.fts_enabled or self.keyword_index is None:
            return []

        query = query.strip()
        if not query:
            return []

        doc_id_score_dict = await self.keyword_index.retrieve(query, limit=limit)
        results = []
        for doc_id, score in doc_id_score_dict.items():
            chunk = self.file_chunks.get(doc_id)
            if chunk:
                results.append(chunk.model_copy(update={"scores": {"keyword": score, "score": score}}))

        return results
