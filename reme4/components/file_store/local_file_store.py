"""In-memory file store with JSONL persistence on close."""

import aiofiles
import numpy as np

from .base_file_store import BaseFileStore
from ..component_registry import R
from ...schema import FileChunk, FileNode
from ...utils import batch_cosine_similarity


@R.register("local")
class LocalFileStore(BaseFileStore):
    """In-memory file store with deferred JSONL persistence."""

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.file_chunks: dict[str, FileChunk] = {}
        self.chunks_path = self.store_path / f"file_chunks_{self.store_version}.jsonl"

    # Lifecycle

    async def _start(self) -> None:
        await super()._start()
        await self.load()
        self.logger.info(f"LocalFileStore '{self.store_name}' ready: {len(self.file_chunks)} chunks")

    async def _close(self) -> None:
        await self.dump()
        self.file_chunks.clear()
        await super()._close()

    async def load(self) -> None:
        """Load chunks from JSONL file into memory."""
        if not self.chunks_path.exists():
            return
        try:
            async with aiofiles.open(self.chunks_path, encoding=self.encoding) as f:
                async for line in f:
                    line = line.strip()
                    if line:
                        chunk = FileChunk.model_validate_json(line)
                        self.file_chunks[chunk.id] = chunk
        except Exception as e:
            self.logger.exception(f"Failed to load {self.chunks_path}: {e}")

    async def dump(self) -> None:
        """Persist chunks to JSONL via atomic rename, then cascade to keyword_index and file_graph."""
        try:
            tmp = self.chunks_path.with_suffix(".tmp")
            async with aiofiles.open(tmp, "w", encoding=self.encoding) as f:
                await f.write("\n".join(c.model_dump_json() for c in self.file_chunks.values()))
            tmp.replace(self.chunks_path)
        except Exception as e:
            self.logger.exception(f"Failed to write {self.chunks_path}: {e}")
        if self.keyword_index:
            await self.keyword_index.dump()
        if self.file_graph:
            await self.file_graph.dump()

    # Base class interface

    async def upsert_file(
        self,
        file: tuple[FileNode, list[FileChunk]] | list[tuple[FileNode, list[FileChunk]]],
    ) -> None:
        if not self.file_graph:
            raise RuntimeError("file_graph is required for upsert_file")
        if isinstance(file, tuple):
            file = [file]

        old_map = {n.path: n for n in await self.file_graph.get_nodes([node.path for node, _ in file])}

        new_nodes: list[FileNode] = []
        needs_embed: list[FileChunk] = []
        keyword_docs: dict[str, str] = {}
        for node, chunks in file:
            old_node: FileNode | None = old_map.get(node.path)
            cached = {}
            if old_node and self.embedding_model:
                for cid in old_node.chunk_ids:
                    old = self.file_chunks.pop(cid, None)
                    if old and old.embedding is not None:
                        cached[cid] = old.embedding

            node.chunk_ids = []
            for c in chunks:
                if self.embedding_model and c.embedding is None:
                    if c.id in cached:
                        c.embedding = cached[c.id]
                    elif c.text:
                        needs_embed.append(c)
                node.chunk_ids.append(c.id)
                self.file_chunks[c.id] = c
                if c.text:
                    keyword_docs[c.id] = c.text
            new_nodes.append(node)

        await self.file_graph.upsert_nodes(new_nodes)
        if needs_embed and self.embedding_model:
            try:
                await self.embedding_model.get_node_embeddings(needs_embed)
            except Exception as e:
                self._disable_embedding(f"upsert: {type(e).__name__}: {e}")
        if self.keyword_index and keyword_docs:
            await self.keyword_index.add_docs(keyword_docs)

    async def delete_by_path(self, path: str | list[str]) -> None:
        if not self.file_graph:
            raise RuntimeError("file_graph is required for delete_by_path")
        if isinstance(path, str):
            path = [path]
        nodes = await self.file_graph.get_nodes(path)
        if not nodes:
            return
        deleted_chunk_ids = [cid for n in nodes for cid in n.chunk_ids]
        for cid in deleted_chunk_ids:
            self.file_chunks.pop(cid, None)
        await self.file_graph.delete_nodes([n.path for n in nodes])
        if self.keyword_index and deleted_chunk_ids:
            await self.keyword_index.delete_docs(deleted_chunk_ids)

    async def clear(self) -> None:
        if not self.file_graph:
            raise RuntimeError("file_graph is required for clear")
        self.file_chunks.clear()
        if self.keyword_index:
            await self.keyword_index.clear()
        await self.file_graph.clear()

    # Search

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if self.embedding_model is None or not query:
            return []

        try:
            query_embedding = await self.embedding_model.get_embedding(query)
        except Exception as e:
            self._disable_embedding(f"search: {type(e).__name__}: {e}")
            return []
        if query_embedding is None:
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
        if not self.keyword_index:
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
