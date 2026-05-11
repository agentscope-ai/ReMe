"""Abstract base class for file stores — minimalist engine surface.

A `FileStore` manages two things, no more:

  * **graph**  — `dict[path, FileNode]` in memory, JSONL-backed by default
                 (sqlite backend overrides for relational storage). Edges
                 live on `FileNode.edges`; there is no separate edge index.
  * **chunks** — `dict[path, list[FileChunk]]`-shaped reads/writes plus
                 vector / keyword search; subclasses implement.

Read APIs (`get_node`, `get_links`, `get_backlinks`, …) are **synchronous**
— they hit the in-memory index rebuilt from persistence on `_start`.
Writes are **async** — they touch persistence first, then update the
in-memory index, so on crash the on-disk view is the source of truth.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from ..base_component import BaseComponent
from ..embedding import BaseEmbeddingModel
from ...enumeration import ComponentEnum
from ...schema import ChunkFilter, FileChunk, FileEdge, FileNode


def _target_stem(raw: str) -> str:
    """Stem of a raw wikilink target — `"topics/X/X.md"` → `"X"`."""
    target = raw.strip().removesuffix(".md")
    return target.rsplit("/", 1)[-1]


class BaseFileStore(BaseComponent):
    """Engine-level store: graph (in-memory + JSON) + chunks (db) + search."""

    component_type = ComponentEnum.FILE_STORE

    def __init__(
        self,
        store_name: str,
        store_path: str | Path,
        embedding_model: str = "default",
        fts_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not re.match(r"^[a-zA-Z0-9_]+$", store_name):
            raise ValueError(
                f"Invalid store name '{store_name}'. "
                f"Only alphanumeric characters and underscores are allowed.",
            )
        self.store_name: str = store_name
        self.store_path: Path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.working_dir = self.app_context.app_config.working_dir if self.app_context is not None else ""


        self._embedding_model_name: str = embedding_model
        self.embedding_model: BaseEmbeddingModel | None = None
        self.vector_enabled: bool = bool(embedding_model)
        self.fts_enabled: bool = fts_enabled
        if not self.vector_enabled and not self.fts_enabled:
            raise ValueError("At least one of embedding_model or fts_enabled must be set.")

        # In-memory state (rebuilt from persistence on _start).
        self._nodes: dict[str, FileNode] = {}
        self._backlinks: dict[str, set[str]] = defaultdict(set)

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        if self._embedding_model_name:
            assert self.app_context is not None, "app_context must be provided"
            models = self.app_context.components.get(ComponentEnum.EMBEDDING_MODEL, {})
            if self._embedding_model_name not in models:
                raise ValueError(f"Embedding model '{self._embedding_model_name}' not found.")
            model = models[self._embedding_model_name]
            if not isinstance(model, BaseEmbeddingModel):
                raise TypeError(f"Expected BaseEmbeddingModel, got {type(model).__name__}")
            self.embedding_model = model
        # Subclass `_start` opens persistence FIRST; this base `_start`
        # runs LAST so persistence is ready by the time we iterate it.
        # Subclasses should `await super()._start()` at the END of their _start.
        await self._reload_nodes()

    async def _close(self) -> None:
        self.embedding_model = None
        self._nodes.clear()
        self._backlinks.clear()

    async def _reload_nodes(self) -> None:
        """Rebuild in-memory indices from persisted nodes. Called on _start."""
        self._nodes.clear()
        self._backlinks.clear()
        node_count = 0
        edge_count = 0
        for node in self._iter_persisted_nodes():
            self._index_node(node)
            node_count += 1
            edge_count += len(node.edges)
        self.logger.info(
            f"[{self.store_name}] Loaded {node_count} nodes, "
            f"{edge_count} edges into in-memory index",
        )

    # -- Embedding (single public entrypoint) ------------------------------

    async def embed(self, texts: list[str]) -> list[list[float] | None] | None:
        """Embed a batch of texts. None on disabled / API failure.

        Returns a list parallel to `texts` whose entries may individually
        be None if the embedding model dropped them. The whole call
        returns None if the store has vector search disabled or the
        embedding API errors out (which also auto-disables vector
        search for the rest of the process).
        """
        if not self.vector_enabled or not texts or self.embedding_model is None:
            return None
        try:
            return await self.embedding_model.get_embeddings(texts)
        except Exception as e:
            self._disable_vector_search(str(e))
            return None

    def _disable_vector_search(self, reason: str = "embedding API error") -> None:
        if self.vector_enabled:
            self.logger.warning(f"[{self.store_name}] Disabling vector search: {reason}")
            self.vector_enabled = False

    # -- Graph CRUD --------------------------------------------------------

    async def upsert_node(self, node: FileNode) -> None:
        """Persist + reindex a node. Replaces any prior entry for `node.path`."""
        # await self._persist_upsert_node(node)
        self._unindex_node(node.path)
        self._index_node(node)

    # async def patch_node(self, path: str, **fields) -> FileNode | None:
    #     """Convenience: `upsert_node(existing.model_copy(update=fields))`."""
    #     existing = self._nodes.get(path)
    #     if existing is None:
    #         return None
    #     updated = existing.model_copy(update=fields)
    #     await self.upsert_node(updated)
    #     return updated

    async def delete_node(self, path: str) -> FileNode | None:
        """Persist deletion + drop from indices. Returns prior node if any."""
        await self._persist_delete_node(path)
        prior = self._unindex_node(path)
        return prior

    async def read_node(self, path: str) -> FileNode | None:
        return self._nodes.get(path)

    # def get_edges(self, path: str) -> list[FileEdge]:
    #     node = self._nodes.get(path)
    #     return list(node.edges) if node is not None else []
    #
    # @property
    # def nodes(self) -> Mapping[str, FileNode]:
    #     return self._nodes
    #
    # def get_paths_by_stem(self, stem: str) -> list[str]:
    #     return sorted(self._stems.get(stem, set()))

    # -- Index maintenance (private) ---------------------------------------

    def _index_node(self, node: FileNode) -> None:
        self._nodes[node.path] = node
        for edge in node.edges:
            self._backlinks[_target_stem(edge.target)].add(node.path)

    def _unindex_node(self, path: str) -> FileNode | None:
        prior = self._nodes.pop(path, None)
        if prior is None:
            return None
        stem = Path(prior.path).stem
        for edge in prior.edges:
            tgt = _target_stem(edge.target)
            self._backlinks.get(tgt, set()).discard(prior.path)
            if not self._backlinks.get(tgt):
                self._backlinks.pop(tgt, None)
        return prior

    # -- Link queries (resolve via reme2.utils.wikilink_resolver) ----------

    # def get_links(self, path: str) -> list[tuple[FileNode, FileEdge]]:
    #     """Files `path` links TO (resolved). One pair per edge, dedup by caller."""
    #     from ...utils.wikilink_resolver import resolve_wikilink
    #
    #     node = self._nodes.get(path)
    #     if node is None:
    #         return []
    #     out: list[tuple[FileNode, FileEdge]] = []
    #     for edge in node.edges:
    #         hit = resolve_wikilink(self, edge.target)
    #         if hit is not None and hit in self._nodes:
    #             out.append((self._nodes[hit], edge))
    #     return out

    # def get_backlinks(self, path: str) -> list[tuple[FileNode, FileEdge]]:
    #     """Files that link TO `path` (resolved). One pair per qualifying edge."""
    #     from ...utils.wikilink_resolver import resolve_wikilink
    #
    #     if path not in self._nodes:
    #         return []
    #     stem = Path(path).stem
    #     out: list[tuple[FileNode, FileEdge]] = []
    #     for src in self._backlinks.get(stem, set()):
    #         src_node = self._nodes.get(src)
    #         if src_node is None:
    #             continue
    #         for edge in src_node.edges:
    #             if resolve_wikilink(self, edge.target) == path:
    #                 out.append((src_node, edge))
    #     return out

    # -- Hot write entry (called by watcher per file change) ---------------

    async def upsert(self, node: FileNode, chunks: list[FileChunk]) -> None:
        """Single fan-out from the watcher's parse pass.

        The parser hands over a fresh node + text-only chunks; the store
        owns the embedding pipeline:
          1. fetch persisted chunks for `node.path`
          2. attach cached embeddings to incoming chunks whose hash matches
          3. embed only the dirty (new-hash) subset
          4. persist node, then chunks
        """
        existing = await self.get_chunks(node.path)
        dirty = self._hash_diff_attach(chunks, existing)
        if dirty:
            await self._embed_chunks(dirty)

        await self.upsert_node(node)
        await self.upsert_chunks(node.path, chunks)

    @staticmethod
    def _hash_diff_attach(
        chunks: list[FileChunk],
        existing_chunks: list[FileChunk] | None,
    ) -> list[FileChunk]:
        """Attach cached embeddings to chunks whose hash already exists.

        Mutates input chunks in place; returns the dirty subset still
        needing embeddings.
        """
        if not existing_chunks:
            return list(chunks)
        cached = {c.hash: c.embedding for c in existing_chunks if c.embedding}
        if not cached:
            return list(chunks)
        dirty: list[FileChunk] = []
        for c in chunks:
            cached_emb = cached.get(c.hash)
            if cached_emb is not None:
                c.embedding = cached_emb
            else:
                dirty.append(c)
        return dirty

    async def _embed_chunks(self, chunks: list[FileChunk]) -> None:
        """In-place attach embeddings via the configured model."""
        if not chunks or self.embedding_model is None or not self.vector_enabled:
            return
        try:
            await self.embedding_model.get_node_embeddings(chunks)
        except Exception as e:
            self.logger.warning(f"embedding chunks failed: {e}")
            self._disable_vector_search(str(e))

    # -- Default node persistence (single sidecar JSONL) -------------------

    @property
    def _nodes_path(self) -> Path:
        return self.store_path / f"{self.store_name}_nodes.jsonl"

    async def _persist_upsert_node(self, node: FileNode) -> None:
        """Default: full rewrite. Backends with relational storage override."""
        snapshot = {**self._nodes, node.path: node}
        self._write_nodes_jsonl(snapshot.values())

    async def _persist_delete_node(self, path: str) -> None:
        """Default: full rewrite, omitting `path`."""
        snapshot = {p: n for p, n in self._nodes.items() if p != path}
        self._write_nodes_jsonl(snapshot.values())

    def _iter_persisted_nodes(self) -> Iterable[FileNode]:
        path = self._nodes_path
        if not path.exists():
            return []
        out: list[FileNode] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(FileNode.model_validate_json(line))
            except Exception as e:
                self.logger.warning(f"Bad row in {path}: {e}")
        return out

    def _write_nodes_jsonl(self, nodes: Iterable[FileNode]) -> None:
        path = self._nodes_path
        content = "\n".join(n.model_dump_json() for n in nodes)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        except Exception as e:
            self.logger.error(f"Failed to write {path}: {e}")
            raise
        finally:
            if tmp.exists():
                tmp.unlink()

    # -- Chunk APIs (subclasses implement) ---------------------------------

    @abstractmethod
    async def upsert_chunks(self, path: str, chunks: list[FileChunk]) -> None:
        """Insert or replace all chunks for a file path. Embeddings pre-attached."""

    @abstractmethod
    async def delete_chunks(self, path: str) -> None:
        """Delete all chunks for a file path."""

    @abstractmethod
    async def get_chunks(self, path: str) -> list[FileChunk]:
        """All chunks for a file path."""

    @abstractmethod
    async def get_chunks_by_paths(self, paths: Iterable[str]) -> list[FileChunk]:
        """Batch fetch chunks across many paths."""


    async def upsert_node(self, node: FileNode):
        ...

    async def delete_node(self, path: str):
        ...

    async def get_node(self, path: str) -> FileNode:
        ...

    async def upsert_chunks(self, path: str, chunks: list[FileChunk]) -> None:
        ...

    async def delete_chunks(self, path: str) -> None:
        ...

    async def get_chunks(self, path: str) -> list[FileChunk]:
        ...

    async def upsert(self, node: FileNode, chunks: list[FileChunk]) -> None:
        ...

    async def delete(self, path: str) -> None:
        ...

    @abstractmethod
    async def vector_search(
        self,
        query: str,
        limit: int,
        search_filter: dict,
    ) -> list[FileChunk]:
        """Vector similarity search."""

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        limit: int,
        search_filter: dict,
    ) -> list[FileChunk]:
        """Full-text/keyword search."""
