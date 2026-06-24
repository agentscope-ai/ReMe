"""Regression tests for LocalFileStore / FaissLocalFileStore consistency."""

# pylint: disable=protected-access

import asyncio
import os
import tempfile

import numpy as np
import pytest

from reme.components.file_store import FaissLocalFileStore, LocalFileStore
from reme.schema import FileChunk, FileNode


class temp_chdir:
    """Temporarily chdir into a test workspace."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


class FakeEmbeddingStore:
    """Small deterministic embedding provider used by file-store tests."""

    dimensions = 2

    def _embed(self, text: str) -> np.ndarray:
        if "beta" in text or "fresh" in text:
            return np.array([0.0, 1.0], dtype=np.float16)
        return np.array([1.0, 0.0], dtype=np.float16)

    async def health_check(self, _timeout: float = 2.0) -> bool:
        """Report the fake embedding service as healthy."""
        return True

    async def get_embedding(self, input_text: str, **_kwargs) -> np.ndarray:
        """Return a deterministic embedding for a single text."""
        return self._embed(input_text)

    async def get_node_embeddings(self, nodes: list[FileChunk], **_kwargs) -> list[FileChunk]:
        """Attach deterministic embeddings to file chunks."""
        for chunk_node in nodes:
            chunk_node.embedding = self._embed(chunk_node.text)
        return nodes


def run(coro):
    """Run an async test body."""
    return asyncio.run(coro)


def node(path: str) -> FileNode:
    """Build a minimal file node."""
    return FileNode(path=path, st_mtime=1.0)


def chunk(chunk_id: str, path: str, text: str, **metadata) -> FileChunk:
    """Build a minimal file chunk."""
    return FileChunk(id=chunk_id, path=path, text=text, start_line=1, end_line=1, metadata=metadata)


def test_keyword_only_upsert_removes_old_chunks_and_docs():
    """Keyword-only upsert removes stale chunks and keyword documents."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_keyword_only", embedding_store="")
            await store.start()

            await store.upsert([(node("note.md"), [chunk("old", "note.md", "obsoleteword only")])])
            assert [c.id for c in await store.keyword_search("obsoleteword", 5, {})] == ["old"]

            await store.upsert([(node("note.md"), [chunk("new", "note.md", "freshword only")])])

            assert "old" not in store.file_chunks
            assert await store.keyword_search("obsoleteword", 5, {}) == []
            assert [c.id for c in await store.keyword_search("freshword", 5, {})] == ["new"]
            await store.close()

    run(go())


def test_load_rebuilds_keyword_index_from_persisted_chunks_when_missing():
    """Loading persisted chunks repairs a missing keyword index."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_keyword_repair", embedding_store="")
            await store.start()

            await store.upsert(
                [
                    (node("a.md"), [chunk("a", "a.md", "uniquerepairword stock")]),
                    (node("b.md"), [chunk("b", "b.md", "work preference")]),
                ],
            )
            await store.dump()

            await store.keyword_index.clear()
            assert not store.keyword_index.index_file.exists()
            assert await store.keyword_search("uniquerepairword", 5, {}) == []

            store.file_chunks.clear()
            await store.load()

            assert store.keyword_index.index_file.exists()
            assert [c.id for c in await store.keyword_search("uniquerepairword", 5, {})] == ["a"]
            await store.close()

    run(go())


def test_same_chunk_id_with_changed_text_gets_new_embedding():
    """Changing a chunk text refreshes its embedding."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_embedding_reuse", embedding_store="")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()

            await store.upsert([(node("note.md"), [chunk("same", "note.md", "alpha text")])])
            assert store.file_chunks["same"].embedding.tolist() == [1.0, 0.0]

            await store.upsert([(node("note.md"), [chunk("same", "note.md", "beta text")])])

            assert store.file_chunks["same"].embedding.tolist() == [0.0, 1.0]
            await store.close()

    run(go())


def test_search_filter_applies_to_vector_and_keyword_results():
    """Search filters apply consistently to vector and keyword results."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_filter", embedding_store="")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()

            await store.upsert(
                [
                    (node("daily/a.md"), [chunk("a", "daily/a.md", "fresh topic", kind="daily")]),
                    (node("resource/b.md"), [chunk("b", "resource/b.md", "fresh topic", kind="resource")]),
                ],
            )

            filt = {"path_prefix": "daily/", "metadata": {"kind": "daily"}}
            assert [c.path for c in await store.vector_search("fresh", 5, filt)] == ["daily/a.md"]
            assert [c.path for c in await store.keyword_search("fresh", 5, filt)] == ["daily/a.md"]
            await store.close()

    run(go())


def test_faiss_rebuilds_stale_sidecar_and_updates_same_id_text():
    """FAISS sidecar rebuilds when persisted rows no longer match chunks."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            try:
                store = FaissLocalFileStore(name="t_faiss", embedding_store="")
            except ImportError:
                pytest.skip("faiss is not installed")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()

            await store.upsert([(node("note.md"), [chunk("same", "note.md", "alpha text")])])
            assert [c.id for c in await store.vector_search("alpha", 5, {})] == ["same"]

            await store.upsert([(node("note.md"), [chunk("same", "note.md", "beta text")])])
            assert [c.id for c in await store.vector_search("beta", 5, {})] == ["same"]
            assert store._id_to_row["same"] == 1

            await store.dump()
            store.file_chunks = {"other": chunk("other", "other.md", "alpha text")}
            store.file_chunks["other"].embedding = np.array([1.0, 0.0], dtype=np.float16)

            assert await store._try_load_sidecar() is False
            store._rebuild_index()
            assert set(store._id_to_row) == {"other"}
            await store.close()

    run(go())
