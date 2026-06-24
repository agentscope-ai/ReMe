"""In-memory file store with compressed JSONL persistence on close."""

from contextlib import suppress

import numpy as np

from .base_file_store import BaseFileStore
from ..component_registry import R
from ..embedding_store import BaseEmbeddingStore
from ..file_graph import BaseFileGraph
from ..keyword_index import BaseKeywordIndex
from ...enumeration import LinkScopeEnum
from ...schema import FileChunk, FileLink, FileNode
from ...utils import batch_cosine_similarity
from ...utils.jsonl_zst import read_jsonl_zst, write_jsonl_zst

CachedEmbedding = tuple[str, np.ndarray]


@R.register("local")
class LocalFileStore(BaseFileStore):
    """In-memory file store with deferred JSONL persistence.

    Composes three subcomponents: ``embedding_store`` for vector retrieval,
    ``keyword_index`` for full-text retrieval, and ``file_graph`` for node / link
    storage. ``file_graph`` is mandatory; at least one of embedding / keyword
    must be present.
    """

    def __init__(
        self,
        embedding_store: str = "default",
        keyword_index: str = "default",
        file_graph: str = "default",
        encoding: str = "utf-8",
        store_version: str = "v1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        from ..embedding_store import LocalEmbeddingStore
        from ..file_graph import LocalFileGraph
        from ..keyword_index import BM25Index

        if not embedding_store and not keyword_index:
            raise ValueError("At least one of embedding_store or keyword_index must be set.")
        if not file_graph:
            raise ValueError("file_graph is required for LocalFileStore.")

        self.embedding_store = self.bind(embedding_store, BaseEmbeddingStore, default_factory=LocalEmbeddingStore)
        self.keyword_index = self.bind(keyword_index, BaseKeywordIndex, default_factory=BM25Index)
        self.file_graph = self.bind(file_graph, BaseFileGraph, default_factory=LocalFileGraph)

        self.encoding = encoding
        self.store_version = store_version
        self.file_chunks: dict[str, FileChunk] = {}
        self.chunks_path = self.component_metadata_path / f"file_chunks_{self.name}_{self.store_version}.jsonl.zst"

    # -- lifecycle ------------------------------------------------------------

    async def _start(self) -> None:
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        await super()._start()
        if self.embedding_store is not None and not await self.embedding_store.health_check():
            self.logger.warning(f"{self.name}: embedding unhealthy, vector disabled")
            self.embedding_store = None
        await self.load()

    async def _close(self) -> None:
        await self.dump()
        self.file_chunks.clear()
        await super()._close()

    def _disable_embedding(self, reason: str) -> None:
        """Drop embedding after a runtime failure; keyword search still works."""
        if self.embedding_store is None:
            return
        self.logger.error(f"{self.name}: embedding disabled, {reason}")
        self.embedding_store = None

    # -- persistence ----------------------------------------------------------

    async def load(self) -> None:
        """Load chunks from the JSONL file into memory; missing file is a no-op."""
        if not self.chunks_path.exists():
            return
        try:
            for line in read_jsonl_zst(self.chunks_path, self.encoding):
                line = line.strip()
                if line:
                    chunk = FileChunk.model_validate_json(line)
                    self.file_chunks[chunk.id] = chunk
            self.logger.info(f"Loaded {len(self.file_chunks)} chunks from {self.chunks_path}")
            await self._sync_keyword_index_from_chunks()
        except Exception as e:
            self.logger.exception(f"Failed to load {self.chunks_path}: {e}")

    async def _sync_keyword_index_from_chunks(self) -> None:
        """Repair keyword index when its persisted state does not match chunks."""
        if not self.keyword_index or not self.file_chunks:
            return

        docs = {cid: chunk.text for cid, chunk in self.file_chunks.items() if chunk.text}
        if not docs:
            return

        expected_ids = set(docs)
        live_ids = None
        with suppress(Exception):
            live_ids = set(getattr(self.keyword_index, "doc_meta", {}).keys())

        if live_ids == expected_ids:
            return

        n_docs = getattr(self.keyword_index, "n_docs", None)
        if live_ids is None and n_docs == len(expected_ids):
            return

        self.logger.warning(f"{self.name}: keyword index mismatch with chunks; rebuilding {len(docs)} docs")
        await self.keyword_index.reset_index(docs)

    async def dump(self) -> None:
        """Atomically rewrite the JSONL, then cascade dump into keyword_index and file_graph."""
        assert self.file_graph is not None
        try:
            write_jsonl_zst(self.chunks_path, (c.model_dump_json() for c in self.file_chunks.values()), self.encoding)
            self.logger.info(f"Saved {len(self.file_chunks)} chunks to {self.chunks_path}")
        except Exception as e:
            self.logger.exception(f"Failed to write {self.chunks_path}: {e}")
        if self.keyword_index:
            await self.keyword_index.dump()
        await self.file_graph.dump()

    # -- CRUD -----------------------------------------------------------------

    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        if not files:
            return
        assert self.file_graph is not None

        old_map = {n.path: n for n in await self.file_graph.get_nodes([node.path for node, _ in files])}
        old_chunk_ids = {cid for n in old_map.values() for cid in n.chunk_ids}
        new_nodes, needs_embed, keyword_docs = self._stage_upsert(files, old_map)

        await self.file_graph.upsert_nodes(new_nodes)
        await self._embed_pending(needs_embed)
        if self.keyword_index and old_chunk_ids:
            await self.keyword_index.delete_docs(list(old_chunk_ids))
        if self.keyword_index and keyword_docs:
            await self.keyword_index.add_docs(keyword_docs)

    def _stage_upsert(
        self,
        files: list[tuple[FileNode, list[FileChunk]]],
        old_map: dict[str, FileNode],
    ) -> tuple[list[FileNode], list[FileChunk], dict[str, str]]:
        """Mutate self.file_chunks for each file and collect the work the I/O step needs:
        new graph nodes, chunks still needing an embedding, and keyword docs to index.
        """
        new_nodes: list[FileNode] = []
        needs_embed: list[FileChunk] = []
        keyword_docs: dict[str, str] = {}
        for node, chunks in files:
            cached = self._evict_prior_chunks(old_map.get(node.path))
            node.chunk_ids = []
            for c in chunks:
                self._reuse_or_queue_embedding(c, cached, needs_embed)
                node.chunk_ids.append(c.id)
                self.file_chunks[c.id] = c
                if c.text:
                    keyword_docs[c.id] = c.text
            new_nodes.append(node)
        return new_nodes, needs_embed, keyword_docs

    def _evict_prior_chunks(self, old_node: FileNode | None) -> dict[str, CachedEmbedding]:
        """Drop chunks for the path being re-upserted; keep their embeddings around so
        a new chunk reusing the same id and text avoids a redundant embedding call.
        """
        cached: dict[str, CachedEmbedding] = {}
        if old_node is None:
            return cached
        for cid in old_node.chunk_ids:
            old = self.file_chunks.pop(cid, None)
            if self.embedding_store and old and old.embedding is not None:
                cached[cid] = (old.text, old.embedding)
        return cached

    def _reuse_or_queue_embedding(
        self,
        chunk: FileChunk,
        cached: dict[str, CachedEmbedding],
        needs_embed: list[FileChunk],
    ) -> None:
        if not self.embedding_store or chunk.embedding is not None:
            return
        if chunk.id in cached and cached[chunk.id][0] == chunk.text:
            chunk.embedding = cached[chunk.id][1]
        elif chunk.text:
            needs_embed.append(chunk)

    async def _embed_pending(self, chunks: list[FileChunk]) -> None:
        if not (chunks and self.embedding_store):
            return
        try:
            await self.embedding_store.get_node_embeddings(chunks)
        except Exception as e:
            self._disable_embedding(f"upsert: {type(e).__name__}: {e}")

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

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        assert self.file_graph is not None
        return await self.file_graph.get_nodes(paths)

    async def get_outlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        assert self.file_graph is not None
        return await self.file_graph.get_outlinks(path, scope)

    async def get_inlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        assert self.file_graph is not None
        return await self.file_graph.get_inlinks(path, scope)

    async def clear(self) -> None:
        assert self.file_graph is not None
        self.file_chunks.clear()
        self.chunks_path.unlink(missing_ok=True)
        if self.keyword_index:
            await self.keyword_index.clear()
        await self.file_graph.clear()

    # -- search ---------------------------------------------------------------

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        if self.embedding_store is None or not query:
            return []

        try:
            query_embedding = await self.embedding_store.get_embedding(query)
        except Exception as e:
            self._disable_embedding(f"search: {type(e).__name__}: {e}")
            return []
        if query_embedding is None:
            return []

        candidates = [
            c
            for c in self.file_chunks.values()
            if c.embedding is not None and self._matches_search_filter(c, search_filter)
        ]
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

        retrieve_limit = limit
        if search_filter:
            retrieve_limit = max(limit, getattr(self.keyword_index, "n_docs", limit))
        doc_id_score_dict = await self.keyword_index.retrieve(query, limit=retrieve_limit)
        results = []
        for doc_id, score in doc_id_score_dict.items():
            chunk = self.file_chunks.get(doc_id)
            if chunk and self._matches_search_filter(chunk, search_filter):
                results.append(chunk.model_copy(update={"scores": {"keyword": score, "score": score}}))
                if len(results) >= limit:
                    break

        return results

    # -- extensions -----------------------------------------------------------

    @staticmethod
    def _as_filter_values(value) -> set:
        if isinstance(value, (list, tuple, set, frozenset)):
            return set(value)
        return {value}

    @classmethod
    def _value_matches(cls, actual, expected) -> bool:
        if isinstance(expected, (list, tuple, set, frozenset)):
            return actual in set(expected)
        return actual == expected

    @classmethod
    def _matches_search_filter(cls, chunk: FileChunk, search_filter: dict | None) -> bool:
        """Conservative post-filter shared by vector and keyword search."""
        if not search_filter:
            return True

        exact_paths = set()
        for key in ("path", "paths"):
            if key in search_filter:
                exact_paths.update(cls._as_filter_values(search_filter[key]))
        if exact_paths and chunk.path not in exact_paths:
            return False

        prefixes = []
        for key in ("path_prefix", "path_prefixes", "prefix", "prefixes"):
            if key in search_filter:
                prefixes.extend(str(v) for v in cls._as_filter_values(search_filter[key]))
        if prefixes and not any(chunk.path.startswith(prefix) for prefix in prefixes):
            return False

        metadata_filter = dict(search_filter.get("metadata") or {})
        reserved = {
            "path",
            "paths",
            "path_prefix",
            "path_prefixes",
            "prefix",
            "prefixes",
            "metadata",
        }
        for key, value in search_filter.items():
            if key not in reserved:
                metadata_filter[key] = value

        return all(cls._value_matches(chunk.metadata.get(key), value) for key, value in metadata_filter.items())

    async def rebuild_links(self) -> None:
        """Rebuild graph links via the underlying file graph."""
        assert self.file_graph is not None
        return await self.file_graph.rebuild_links()
