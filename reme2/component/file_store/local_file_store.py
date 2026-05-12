"""In-memory file store with JSONL persistence on close."""

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
        self._encoding = encoding
        self._nodes: dict[str, FileNode] = {}
        self._chunks: dict[str, FileChunk] = {}
        self._nodes_file = self.store_path / "nodes.jsonl"
        self._chunks_file = self.store_path / "chunks.jsonl"

    # Lifecycle

    async def _start(self) -> None:
        self._load(self._nodes_file, self._nodes, FileNode, "path")
        self._load(self._chunks_file, self._chunks, FileChunk, "id")
        await super()._start()
        self.logger.info(
            f"LocalFileStore '{self.store_name}' ready: "
            f"{len(self._nodes)} nodes, {len(self._chunks)} chunks"
        )

    async def _close(self) -> None:
        self._dump(self._nodes_file, self._nodes.values())
        self._dump(self._chunks_file, self._chunks.values())
        self._nodes.clear()
        self._chunks.clear()
        await super()._close()

    def _load(self, file: Path, target: dict, model: type[BaseModel], key: str) -> None:
        if not file.exists():
            return
        target.clear()
        try:
            for line in file.read_text(encoding=self._encoding).splitlines():
                if line.strip():
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

    # Node operations

    async def upsert_node(self, node: FileNode) -> None:
        self._nodes[node.path] = node

    async def get_node(self, path: str) -> FileNode | None:
        return self._nodes.get(path)

    async def delete_node(self, path: str) -> None:
        self._nodes.pop(path, None)

    # Chunk operations

    async def upsert_chunks(self, path: str, chunks: list[FileChunk]) -> None:
        existing = await self.get_chunks(path)
        cached = {c.hash: c.embedding for c in existing if c.embedding}

        await self.delete_chunks(path)
        if not chunks:
            return

        needs_embed: list[FileChunk] = []
        for c in chunks:
            if c.embedding:
                continue
            if c.hash in cached:
                c.embedding = cached[c.hash]
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

    async def get_chunks(self, path: str) -> list[FileChunk]:
        chunks = [c for c in self._chunks.values() if c.path == path]
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    async def delete_chunks(self, path: str) -> None:
        stale = [cid for cid, c in self._chunks.items() if c.path == path]
        for cid in stale:
            del self._chunks[cid]

    # Search

    async def vector_search(
            self, query: str, limit: int, search_filter: dict,
    ) -> list[FileChunk]:
        if not self.vector_enabled or not query:
            return []
        embeddings = await self.get_embeddings([query])
        if not embeddings or not embeddings[0]:
            return []

        candidates = [c for c in self._chunks.values() if c.embedding]
        if not candidates:
            return []

        chunk_embs = np.array([c.embedding for c in candidates])
        similarities = batch_cosine_similarity(np.array([embeddings[0]]), chunk_embs)[0]

        results = [
            c.model_copy(update={"scores": {"vector": float(s), "score": float(s)}})
            for c, s in zip(candidates, similarities)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def keyword_search(
            self, query: str, limit: int, search_filter: dict,
    ) -> list[FileChunk]:
        if not self.fts_enabled or not query.split():
            return []

        results = [
            c.model_copy(update={"scores": {"keyword": s, "score": s}})
            for c in self._chunks.values()
            if (s := self._keyword_score(query, c.text)) > 0
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # Helpers

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
