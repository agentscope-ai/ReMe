"""Small shared constructors for local, FAISS, and zvec store contract tests."""

# pylint: disable=protected-access

import asyncio
import time

import numpy as np
import pytest

from reme.components.file_store import FaissLocalFileStore, LocalFileStore, ZvecLocalFileStore
from reme.schema import FileChunk, FileNode


class FakeEmbeddingStore:
    """Small deterministic embedding provider used by file-store tests."""

    dimensions = 2
    max_batch_size = 10
    is_healthy = True

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

    async def dump(self) -> None:
        """Persist no state for the in-memory fake."""


def node(path: str) -> FileNode:
    """Build a minimal file node."""
    return FileNode(path=path, st_mtime=1.0)


def chunk(chunk_id: str, path: str, text: str, **metadata) -> FileChunk:
    """Build a minimal file chunk."""
    return FileChunk(id=chunk_id, path=path, text=text, start_line=1, end_line=1, metadata=metadata)


async def _seed_unembedded_chunk(store: LocalFileStore, chunk_id: str, path: str, text: str) -> None:
    """Attach one chunk without a vector, keeping the graph invariant intact."""
    store.file_chunks[chunk_id] = chunk(chunk_id, path, text)
    file_node = node(path)
    file_node.chunk_ids = [chunk_id]
    await store.file_graph.upsert_nodes([file_node])


def _new_faiss_store(name, **kwargs):
    """Construct a FAISS store when the optional backend is installed."""
    try:
        store = FaissLocalFileStore(name=name, embedding_store="", **kwargs)
    except ImportError:
        pytest.skip("faiss is not installed")
    return store


def _new_zvec_store(name, **kwargs):
    """Construct a zvec store with embedding disabled at bind time."""
    try:
        store = ZvecLocalFileStore(name=name, embedding_store="", **kwargs)
    except ImportError:
        pytest.skip("zvec is not installed")
    return store


async def _settle_reindex(store, timeout=5.0):
    """Wait until no async reindex is pending or in flight."""
    deadline = time.monotonic() + timeout
    while store._reindex_event.is_set() or store._reindex_busy:
        if time.monotonic() > deadline:
            raise AssertionError("async reindex did not settle in time")
        await asyncio.sleep(0.005)
