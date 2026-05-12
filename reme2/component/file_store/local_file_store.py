"""Pure-Python in-memory store with on-close JSONL persistence.

Runtime model:
  * All state lives in two dicts (`_nodes`, `_chunks`) — every read /
    write is a dict op, no per-call I/O.
  * `_start` rehydrates from `{store_name}_nodes.jsonl` and
    `{store_name}_chunks.jsonl` under `store_path`.
  * `_close` flushes the full in-memory snapshot back to the same
    sidecar files (atomic via tmp + replace).

Trade-off: lower write latency, but a hard crash drops anything since
the last clean shutdown.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .base_file_store import BaseFileStore
from ..component_registry import R
from ...schema import FileChunk, FileNode
from ...utils import batch_cosine_similarity


@R.register("local")
class LocalFileStore(BaseFileStore):
    """In-memory file store with deferred JSONL persistence."""

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self._encoding: str = encoding
        self._nodes: dict[str, FileNode] = {}
        self._chunks: dict[str, FileChunk] = {}
        self._nodes_file: Path = self.store_path / f"{self.store_name}_nodes.jsonl"
        self._chunks_file: Path = self.store_path / f"{self.store_name}_chunks.jsonl"

    # -- Lifecycle ---------------------------------------------------------

    async def _start(self) -> None:
        self._load(self._nodes_file, self._nodes, FileNode, key="path")
        self._load(self._chunks_file, self._chunks, FileChunk, key="id")
        await super()._start()
        self.logger.info(
            f"LocalFileStore '{self.store_name}' ready: "
            f"{len(self._nodes)} nodes, {len(self._chunks)} chunks",
        )

    async def _close(self) -> None:
        self._dump(self._nodes_file, self._nodes.values())
        self._dump(self._chunks_file, self._chunks.values())
        self._nodes.clear()
        self._chunks.clear()
        await super()._close()

    def _load(
        self, file: Path, target: dict, model: type[BaseModel], key: str,
    ) -> None:
        if not file.exists():
            return
        target.clear()
        try:
            for line in file.read_text(encoding=self._encoding).splitlines():
                if not line.strip():
                    continue
                obj = model.model_validate_json(line)
                target[getattr(obj, key)] = obj
        except Exception as e:
            self.logger.warning(f"Failed to load {file}: {e}")

    def _dump(self, file: Path, items) -> None:
        try:
            content = "\n".join(o.model_dump_json() for o in items)
            tmp = file.with_suffix(".tmp")
            tmp.write_text(content, encoding=self._encoding)
            tmp.replace(file)
        except Exception as e:
            self.logger.error(f"Failed to write {file}: {e}")

    # -- Node CRUD ---------------------------------------------------------

    async def upsert_node(self, node: FileNode) -> None:
        self._nodes[node.path] = node

    async def delete_node(self, path: str) -> None:
        self._nodes.pop(path, None)

    async def get_node(self, path: str) -> FileNode | None:
        return self._nodes.get(path)

    # -- Chunk CRUD --------------------------------------------------------

    async def upsert_chunks(self, path: str, chunks: list[FileChunk]) -> None:
        """Replace all chunks for `path`.

        Hash-diff: chunks whose `hash` matches a persisted one inherit
        the cached embedding; only the new-hash subset hits the embedding
        API.
        """
        existing = await self.get_chunks(path)
        cached = {c.hash: c.embedding for c in existing if c.embedding is not None}

        await self.delete_chunks(path)
        if not chunks:
            return

        needs_embed: list[FileChunk] = []
        for c in chunks:
            if c.embedding is not None:
                continue
            cached_emb = cached.get(c.hash)
            if cached_emb is not None:
                c.embedding = cached_emb
            elif c.text:
                needs_embed.append(c)

        if needs_embed:
            embeddings = await self.embed([c.text for c in needs_embed])
            if embeddings is not None:
                for c, emb in zip(needs_embed, embeddings):
                    if emb is not None:
                        c.embedding = emb

        for c in chunks:
            self._chunks[c.id] = c

    async def delete_chunks(self, path: str) -> None:
        stale = [cid for cid, c in self._chunks.items() if c.path == path]
        for cid in stale:
            del self._chunks[cid]

    async def get_chunks(self, path: str) -> list[FileChunk]:
        chunks = [c for c in self._chunks.values() if c.path == path]
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    # -- Search ------------------------------------------------------------

    async def vector_search(
        self, query: str, limit: int, search_filter: dict,
    ) -> list[FileChunk]:
        if not self.vector_enabled or not query:
            return []
        embeddings = await self.embed([query])
        if not embeddings or embeddings[0] is None:
            return []
        query_emb = embeddings[0]

        # TODO: honor `search_filter` (paths / exclude_paths / tags / ...).
        candidates = [c for c in self._chunks.values() if c.embedding]
        if not candidates:
            return []

        chunk_embs = np.array([c.embedding for c in candidates])
        similarities = batch_cosine_similarity(
            np.array([query_emb]), chunk_embs,
        )[0]

        results: list[FileChunk] = []
        for c, sim in zip(candidates, similarities):
            results.append(c.model_copy(
                update={"scores": {"vector": float(sim), "score": float(sim)}},
            ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def keyword_search(
        self, query: str, limit: int, search_filter: dict,
    ) -> list[FileChunk]:
        if not self.fts_enabled or not query.split():
            return []

        # TODO: honor `search_filter` (paths / exclude_paths / tags / ...).
        results: list[FileChunk] = []
        for c in self._chunks.values():
            score = self._keyword_score(query, c.text)
            if score > 0:
                results.append(c.model_copy(
                    update={"scores": {"keyword": score, "score": score}},
                ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _keyword_score(query: str, text: str) -> float:
        """Word-overlap score with phrase bonus. Range [0, 1]."""
        words = query.split()
        if not words or not text:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for w in words if w.lower() in text_lower)
        if matches == 0:
            return 0.0
        base = matches / len(words)
        if len(words) > 1 and query.lower() in text_lower:
            base = min(1.0, base + 0.2)
        return base
