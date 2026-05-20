"""In-memory file store with JSONL persistence on close."""

import aiofiles
import numpy as np

from .base_file_store import BaseFileStore
from ..component_registry import R
from ..embedding import BaseEmbeddingModel
from ..file_graph import BaseFileGraph
from ..keyword_index import BaseKeywordIndex
from ...schema import FileChunk, FileLink, FileNode
from ...utils import batch_cosine_similarity


@R.register("local")
class LocalFileStore(BaseFileStore):
    """In-memory file store with deferred JSONL persistence.

    Composes three subcomponents: ``embedding_model`` for vector retrieval,
    ``keyword_index`` for full-text retrieval, and ``file_graph`` for node / link
    storage. ``file_graph`` is mandatory; at least one of embedding / keyword
    must be present.
    """

    def __init__(
        self,
        embedding_model: str = "default",
        keyword_index: str = "default",
        file_graph: str = "default",
        encoding: str = "utf-8",
        **kwargs,
    ):
        super().__init__(**kwargs)
        from ..embedding import OpenAIEmbeddingModel
        from ..file_graph import LocalFileGraph
        from ..keyword_index import BM25Index

        if not embedding_model and not keyword_index:
            raise ValueError("At least one of embedding_model or keyword_index must be set.")
        if not file_graph:
            raise ValueError("file_graph is required for LocalFileStore.")

        self.embedding_model = self.bind(embedding_model, BaseEmbeddingModel, default_factory=OpenAIEmbeddingModel)
        self.keyword_index = self.bind(keyword_index, BaseKeywordIndex, default_factory=BM25Index)
        self.file_graph = self.bind(file_graph, BaseFileGraph, default_factory=LocalFileGraph)

        self.encoding = encoding
        self.file_chunks: dict[str, FileChunk] = {}
        self.chunks_path = self.store_path / f"file_chunks_{self.store_version}.jsonl"

    # Lifecycle

    async def _start(self) -> None:
        await super()._start()
        if self.embedding_model is not None and not await self.embedding_model.health_check():
            self.logger.warning(f"{self.store_name}: embedding unhealthy, vector disabled")
            self.embedding_model = None
        await self.load()

    async def _close(self) -> None:
        await self.dump()
        self.file_chunks.clear()
        await super()._close()

    def _disable_embedding(self, reason: str) -> None:
        """Drop embedding after a runtime failure; keyword search still works."""
        if self.embedding_model is None:
            return
        self.logger.error(f"{self.store_name}: embedding disabled, {reason}")
        self.embedding_model = None

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
            self.logger.info(f"Loaded {len(self.file_chunks)} chunks from {self.chunks_path}")
        except Exception as e:
            self.logger.exception(f"Failed to load {self.chunks_path}: {e}")

    async def dump(self) -> None:
        """Persist chunks to JSONL via atomic rename, then cascade to keyword_index and file_graph."""
        assert self.file_graph is not None
        try:
            tmp = self.chunks_path.with_suffix(".tmp")
            async with aiofiles.open(tmp, "w", encoding=self.encoding) as f:
                await f.write("\n".join(c.model_dump_json() for c in self.file_chunks.values()))
            tmp.replace(self.chunks_path)
            self.logger.info(f"Saved {len(self.file_chunks)} chunks to {self.chunks_path}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self.chunks_path}: {e}")
        if self.keyword_index:
            await self.keyword_index.dump()
        await self.file_graph.dump()

    # Write

    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        if not files:
            return
        assert self.file_graph is not None

        old_map = {n.path: n for n in await self.file_graph.get_nodes([node.path for node, _ in files])}

        new_nodes: list[FileNode] = []
        needs_embed: list[FileChunk] = []
        keyword_docs: dict[str, str] = {}
        for node, chunks in files:
            old_node: FileNode | None = old_map.get(node.path)
            cached: dict = {}
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

    async def delete(self, path: str | list[str]) -> None:
        assert self.file_graph is not None
        paths = [path] if isinstance(path, str) else path
        nodes: list[FileNode] = await self.file_graph.get_nodes(paths)
        if not nodes:
            return
        deleted_chunk_ids = [cid for n in nodes for cid in n.chunk_ids]
        for cid in deleted_chunk_ids:
            self.file_chunks.pop(cid, None)
        await self.file_graph.delete_nodes([str(n.path) for n in nodes])
        if self.keyword_index and deleted_chunk_ids:
            await self.keyword_index.delete_docs(deleted_chunk_ids)

    async def clear(self) -> None:
        assert self.file_graph is not None
        self.file_chunks.clear()
        self.chunks_path.unlink(missing_ok=True)
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

    # Graph queries

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        assert self.file_graph is not None
        return await self.file_graph.get_nodes(paths)

    async def get_outlinks(self, path: str) -> list[FileLink]:
        assert self.file_graph is not None
        return await self.file_graph.get_outlinks(path)

    async def get_inlinks(self, path: str) -> list[FileLink]:
        assert self.file_graph is not None
        return await self.file_graph.get_inlinks(path)

    async def rebuild_links(self) -> None:
        assert self.file_graph is not None
        return await self.file_graph.rebuild_links()
