"""Regression tests for LocalFileStore / FaissLocalFileStore consistency."""

# pylint: disable=protected-access

import asyncio
import base64
import json
import os
import tempfile
import threading
import time

import numpy as np
import pytest

from reme.components.file_store import FaissLocalFileStore, LocalFileStore
from reme.components.file_store import local_file_store as local_file_store_module
from reme.schema import FileChunk, FileNode
from reme.utils.jsonl_zst import read_jsonl_zst, write_jsonl_zst


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
    max_batch_size = 10

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


class CountingFakeEmbeddingStore(FakeEmbeddingStore):
    """Fake embedding store that records node backfill requests."""

    def __init__(self):
        self.node_embedding_calls: list[list[str]] = []

    async def get_node_embeddings(self, nodes: list[FileChunk], **_kwargs) -> list[FileChunk]:
        self.node_embedding_calls.append([node.id for node in nodes])
        return await super().get_node_embeddings(nodes, **_kwargs)


class UnhealthyCountingEmbeddingStore(CountingFakeEmbeddingStore):
    """Fake embedding store that fails the backfill health gate."""

    async def health_check(self, _timeout: float = 2.0) -> bool:
        return False


class HealthCountingEmbeddingStore(FakeEmbeddingStore):
    """Fake provider that records eager health checks."""

    def __init__(self):
        self.health_calls = 0

    async def health_check(self, _timeout: float = 2.0) -> bool:
        self.health_calls += 1
        return True


class BlockingEmbeddingStore(FakeEmbeddingStore):
    """Fake provider that proves startup does not await remote backfill."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def get_node_embeddings(self, nodes: list[FileChunk], **kwargs) -> list[FileChunk]:
        self.started.set()
        await self.release.wait()
        return await super().get_node_embeddings(nodes, **kwargs)


class WrongDimEmbeddingStore(FakeEmbeddingStore):
    """Fake embedding store that returns vectors with the wrong dimension."""

    async def get_embedding(self, input_text: str, **_kwargs) -> np.ndarray:
        return np.array([1.0], dtype=np.float16)

    async def get_node_embeddings(self, nodes: list[FileChunk], **_kwargs) -> list[FileChunk]:
        for chunk_node in nodes:
            chunk_node.embedding = np.array([1.0], dtype=np.float16)
        return nodes


class CountOnlyKeywordIndex:
    """Keyword backend that knows its size but cannot expose document IDs."""

    def __init__(self, n_docs: int):
        self.n_docs = n_docs
        self.reset_docs = None

    @property
    def document_ids(self):
        """Signal that exact live IDs are unavailable."""
        raise NotImplementedError

    async def reset_index(self, docs):
        """Record the documents requested for rebuilding."""
        self.reset_docs = docs


def run(coro):
    """Run an async test body."""
    return asyncio.run(coro)


def node(path: str) -> FileNode:
    """Build a minimal file node."""
    return FileNode(path=path, st_mtime=1.0)


def chunk(chunk_id: str, path: str, text: str, **metadata) -> FileChunk:
    """Build a minimal file chunk."""
    return FileChunk(id=chunk_id, path=path, text=text, start_line=1, end_line=1, metadata=metadata)


async def set_chunks_with_graph(store: LocalFileStore, chunks: dict[str, FileChunk]) -> None:
    """Seed a graph/chunk snapshot that satisfies the persistence invariant."""
    store.file_chunks = chunks
    chunk_ids_by_path: dict[str, list[str]] = {}
    for chunk_node in chunks.values():
        chunk_ids_by_path.setdefault(chunk_node.path, []).append(chunk_node.id)
    nodes = []
    for path, chunk_ids in chunk_ids_by_path.items():
        file_node = node(path)
        file_node.chunk_ids = chunk_ids
        nodes.append(file_node)
    await store.file_graph.upsert_nodes(nodes)


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


def test_start_does_not_health_check_embedding_without_backfill():
    """Hot startup keeps local vector retrieval independent of provider health."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_lazy_embedding_health", embedding_store="")
            embedding_store = HealthCountingEmbeddingStore()
            store.embedding_store = embedding_store
            await store.start()

            assert embedding_store.health_calls == 0
            assert store.embedding_store is embedding_store
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


def test_load_clears_graph_when_persisted_chunks_are_missing():
    """A surviving graph must not hide a missing chunk store from reindex."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            seed = LocalFileStore(name="t_missing_chunks", embedding_store="")
            await seed.start()
            indexed_node = node("memory.md")
            await seed.upsert(
                [(indexed_node, [chunk("memory-chunk", "memory.md", "remember this")])],
            )
            await seed.close()

            seed.chunks_path.unlink()
            store = LocalFileStore(name="t_missing_chunks", embedding_store="")
            await store.start()

            assert store.file_chunks == {}
            assert await store.get_nodes() == []
            assert set(store.keyword_index.document_ids) == set()
            await store.close()

    run(go())


def test_load_clears_graph_and_chunks_when_chunk_sets_partially_diverge():
    """Missing and orphaned chunks invalidate the atomic derived snapshot."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_torn_chunks", embedding_store="")
            await store.start()
            indexed_node = node("memory.md")
            indexed_node.chunk_ids = ["kept", "missing"]
            await store.file_graph.upsert_nodes([indexed_node])
            store.file_chunks = {
                "kept": chunk("kept", "memory.md", "kept text"),
                "orphaned": chunk("orphaned", "old.md", "orphaned text"),
            }

            repaired = await store._repair_graph_chunk_consistency()

            assert repaired is True
            assert store.file_chunks == {}
            assert await store.get_nodes() == []
            await store.close()

    run(go())


def test_load_clears_stale_keyword_index_when_chunks_are_empty():
    """An empty chunk store is still an exact state BM25 must mirror."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            seed = LocalFileStore(name="t_empty_chunk_keyword", embedding_store="")
            await seed.start()
            await seed.keyword_index.add_docs({"stale": "stale keyword document"})
            await seed.close()

            store = LocalFileStore(name="t_empty_chunk_keyword", embedding_store="")
            await store.start()

            assert store.file_chunks == {}
            assert set(store.keyword_index.document_ids) == set()
            assert not store.keyword_index.index_file.exists()
            await store.close()

    run(go())


def test_keyword_sync_rebuilds_when_backend_only_exposes_matching_count():
    """Matching counts cannot prove that a backend contains the expected IDs."""

    async def go():
        store = LocalFileStore(name="t_count_only_keyword", embedding_store="")
        store.file_chunks = {
            "expected": chunk("expected", "expected.md", "expected content"),
        }
        keyword_index = CountOnlyKeywordIndex(n_docs=1)
        store.keyword_index = keyword_index

        await store._sync_keyword_index_from_chunks()

        assert keyword_index.reset_docs == {"expected": "expected content"}

    run(go())


def test_keyword_sync_rebuilds_in_progress_batches(monkeypatch):
    """Foreground keyword repair uses bounded batches suitable for progress reporting."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_keyword_progress", embedding_store="")
            await store.start()
            store.file_chunks = {str(index): chunk(str(index), f"{index}.md", f"content {index}") for index in range(5)}
            await store.keyword_index.clear()

            batch_sizes = []
            original_add_docs = store.keyword_index.add_docs

            async def recording_add_docs(docs):
                batch_sizes.append(len(docs))
                await original_add_docs(docs)

            monkeypatch.setattr(local_file_store_module, "_KEYWORD_REBUILD_BATCH_SIZE", 2)
            monkeypatch.setattr(store.keyword_index, "add_docs", recording_add_docs)

            await store._sync_keyword_index_from_chunks()

            assert batch_sizes == [2, 2, 1]
            assert set(store.keyword_index.document_ids) == set(store.file_chunks)
            await store.close()

    run(go())


def test_chunk_persistence_uses_compact_embedding_and_round_trips():
    """Chunk persistence avoids JSON float lists while preserving float16 vectors."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_compact_embedding", embedding_store="")
            await store.start()
            original = chunk("a", "a.md", "alpha text", source="test")
            original.embedding = np.array([0.25, -1.5, 3.0], dtype=np.float16)
            await set_chunks_with_graph(store, {original.id: original})
            await store.dump()

            payload = json.loads(next(read_jsonl_zst(store.chunks_path)))
            assert "embedding" not in payload
            assert isinstance(payload["_embedding_f16_b64"], str)
            assert base64.b64decode(payload["_embedding_f16_b64"]) == original.embedding.astype("<f2").tobytes()

            store.file_chunks.clear()
            await store.load()
            restored = store.file_chunks[original.id]
            np.testing.assert_array_equal(restored.embedding, original.embedding)
            assert restored.embedding.dtype == np.float16
            assert restored.metadata == {"source": "test"}
            await store.close()

    run(go())


def test_vector_search_batches_candidates_and_preserves_stable_ties(monkeypatch):
    """Local vector search limits matrix size and retains insertion order for ties."""

    async def go():
        store = LocalFileStore(name="t_vector_batches", embedding_store="")
        store.embedding_store = FakeEmbeddingStore()
        for index in range(5):
            candidate = chunk(str(index), f"{index}.md", "alpha")
            candidate.embedding = np.array([1.0, 0.0], dtype=np.float16)
            store.file_chunks[candidate.id] = candidate

        batch_sizes = []
        original_similarity = local_file_store_module.batch_cosine_similarity

        def recording_similarity(query, matrix):
            batch_sizes.append(len(matrix))
            return original_similarity(query, matrix)

        monkeypatch.setattr(local_file_store_module, "_VECTOR_SEARCH_BATCH_SIZE", 2)
        monkeypatch.setattr(local_file_store_module, "batch_cosine_similarity", recording_similarity)

        results = await store.vector_search("alpha", 3, {})

        assert batch_sizes == [2, 2, 1]
        assert [result.id for result in results] == ["0", "1", "2"]

    run(go())


def test_chunk_persistence_loads_legacy_json_embedding_list():
    """Existing indexes with JSON float-list embeddings remain readable."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_legacy_embedding", embedding_store="")
            await store.start()
            original = chunk("legacy", "legacy.md", "legacy text")
            original.embedding = np.array([0.5, 1.5], dtype=np.float16)
            write_jsonl_zst(store.chunks_path, [original.model_dump_json()])
            await set_chunks_with_graph(store, {})
            legacy_node = node("legacy.md")
            legacy_node.chunk_ids = [original.id]
            await store.file_graph.upsert_nodes([legacy_node])

            await store.load()
            restored = store.file_chunks[original.id]
            np.testing.assert_array_equal(restored.embedding, original.embedding)
            assert restored.embedding.dtype == np.float16
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


def test_load_backfills_missing_embeddings_from_persisted_chunks():
    """Startup backfills old chunks in the background and persists vectors."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_embedding_backfill", embedding_store="")
            await store.start()
            await store.upsert(
                [
                    (node("a.md"), [chunk("a", "a.md", "alpha text")]),
                    (node("b.md"), [chunk("b", "b.md", "fresh beta text")]),
                ],
            )
            await store.close()

            store = LocalFileStore(name="t_embedding_backfill", embedding_store="")
            store.embedding_store = FakeEmbeddingStore()
            await store.start()
            await store._embedding_backfill_task

            assert store.file_chunks["a"].embedding.tolist() == [1.0, 0.0]
            assert store.file_chunks["b"].embedding.tolist() == [0.0, 1.0]
            await store.close()

            store = LocalFileStore(name="t_embedding_backfill", embedding_store="")
            await store.start()
            assert store.file_chunks["a"].embedding.tolist() == [1.0, 0.0]
            assert store.file_chunks["b"].embedding.tolist() == [0.0, 1.0]
            await store.close()

    run(go())


def test_start_does_not_wait_for_embedding_backfill():
    """Remote embedding repair runs after the file store becomes ready."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            seed = LocalFileStore(name="t_background_embedding", embedding_store="")
            await seed.start()
            await set_chunks_with_graph(seed, {"a": chunk("a", "a.md", "alpha text")})
            await seed.dump()
            await seed.close()

            store = LocalFileStore(name="t_background_embedding", embedding_store="")
            fake = BlockingEmbeddingStore()
            store.embedding_store = fake
            await store.start()

            await asyncio.wait_for(fake.started.wait(), timeout=1)
            assert store.is_started
            assert store.file_chunks["a"].embedding is None

            fake.release.set()
            await store._embedding_backfill_task
            assert store.file_chunks["a"].embedding.tolist() == [1.0, 0.0]
            await store.close()

    run(go())


def test_background_embedding_backfill_uses_provider_batch_size():
    """Embedding repair reports progress over the provider's bounded batches."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            seed = LocalFileStore(name="t_embedding_batches", embedding_store="")
            await seed.start()
            await set_chunks_with_graph(
                seed,
                {str(index): chunk(str(index), f"{index}.md", f"content {index}") for index in range(5)},
            )
            await seed.dump()
            await seed.close()

            store = LocalFileStore(name="t_embedding_batches", embedding_store="")
            fake = CountingFakeEmbeddingStore()
            fake.max_batch_size = 2
            store.embedding_store = fake
            await store.start()
            await store._embedding_backfill_task

            assert [len(batch) for batch in fake.node_embedding_calls] == [2, 2, 1]
            assert all(chunk.embedding is not None for chunk in store.file_chunks.values())
            await store.close()

    run(go())


def test_load_skips_backfill_when_embedding_health_check_fails():
    """Background backfill disables embeddings before batching when the provider is unhealthy."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_embedding_backfill_unhealthy", embedding_store="")
            await store.start()
            await set_chunks_with_graph(store, {"a": chunk("a", "a.md", "alpha text")})
            await store.dump()
            await store.close()

            store = LocalFileStore(name="t_embedding_backfill_unhealthy", embedding_store="")
            fake = UnhealthyCountingEmbeddingStore()
            store.embedding_store = fake
            await store.start()
            await store._embedding_backfill_task

            assert not fake.node_embedding_calls
            assert store.embedding_store is None
            assert store.file_chunks["a"].embedding is None
            await store.close()

    run(go())


def test_load_reembeds_persisted_chunks_with_stale_embedding_dimensions():
    """Loading persisted chunks re-embeds vectors that do not match current dimensions."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_embedding_stale_dim", embedding_store="")
            await store.start()
            stale = chunk("a", "a.md", "alpha text")
            stale.embedding = np.array([1.0], dtype=np.float16)
            await set_chunks_with_graph(store, {"a": stale})
            await store.dump()
            await store.close()

            store = LocalFileStore(name="t_embedding_stale_dim", embedding_store="")
            fake = CountingFakeEmbeddingStore()
            store.embedding_store = fake
            await store.start()
            await store._embedding_backfill_task

            assert fake.node_embedding_calls == [["a"]]
            assert store.file_chunks["a"].embedding.tolist() == [1.0, 0.0]
            await store.close()

    run(go())


def test_drop_stale_embedding_noops_without_embedding_store():
    """The helper should not clear embeddings when vector search is disabled."""

    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = LocalFileStore(name="t_embedding_no_store_drop", embedding_store="")
        stale = chunk("a", "a.md", "alpha text")
        stale.embedding = np.array([1.0], dtype=np.float16)

        assert store._drop_stale_embedding(stale, "test") is False
        assert stale.embedding.tolist() == [1.0]


def test_upsert_does_not_reuse_cached_embedding_with_stale_dimensions():
    """Re-upsert queues a fresh embedding when cached same-text vector has old dimensions."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_embedding_stale_reuse", embedding_store="")
            await store.start()
            fake = CountingFakeEmbeddingStore()
            store.embedding_store = fake

            await store.upsert([(node("note.md"), [chunk("same", "note.md", "alpha text")])])
            store.file_chunks["same"].embedding = np.array([1.0], dtype=np.float16)
            await store.file_graph.upsert_nodes([FileNode(path="note.md", st_mtime=1.0, chunk_ids=["same"])])

            await store.upsert([(node("note.md"), [chunk("same", "note.md", "alpha text")])])

            assert fake.node_embedding_calls == [["same"], ["same"]]
            assert store.file_chunks["same"].embedding.tolist() == [1.0, 0.0]
            await store.close()

    run(go())


def test_upsert_drops_wrong_dimension_from_custom_embedding_store():
    """Wrong-dimensional embeddings from custom stores are not persisted on chunks."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_embedding_wrong_dim_custom", embedding_store="")
            await store.start()
            store.embedding_store = WrongDimEmbeddingStore()

            await store.upsert([(node("note.md"), [chunk("a", "note.md", "alpha text")])])

            assert store.file_chunks["a"].embedding is None
            assert await store.vector_search("alpha", 5, {}) == []
            assert store.embedding_store is None
            await store.close()

    run(go())


def test_upsert_reembeds_prefilled_chunk_with_stale_dimension():
    """Incoming chunks with stale embeddings are re-embedded before persistence."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_embedding_prefilled_stale", embedding_store="")
            await store.start()
            fake = CountingFakeEmbeddingStore()
            store.embedding_store = fake
            prefilled = chunk("a", "note.md", "alpha text")
            prefilled.embedding = np.array([1.0], dtype=np.float16)

            await store.upsert([(node("note.md"), [prefilled])])

            assert fake.node_embedding_calls == [["a"]]
            assert store.file_chunks["a"].embedding.tolist() == [1.0, 0.0]
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


def test_faiss_concurrent_dumps_are_serialized(monkeypatch):
    """Concurrent persistence must not interleave writes to the FAISS sidecar pair."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            try:
                store = FaissLocalFileStore(name="t_faiss_dump_lock", embedding_store="")
            except ImportError:
                pytest.skip("faiss is not installed")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()

            first_started = asyncio.Event()
            release_first = asyncio.Event()
            active_writers = 0
            max_active_writers = 0
            write_count = 0
            original_write_sidecar = store._write_sidecar

            async def blocking_write_sidecar():
                nonlocal active_writers, max_active_writers, write_count
                active_writers += 1
                max_active_writers = max(max_active_writers, active_writers)
                write_count += 1
                if write_count == 1:
                    first_started.set()
                    await release_first.wait()
                active_writers -= 1

            monkeypatch.setattr(store, "_write_sidecar", blocking_write_sidecar)

            first = asyncio.create_task(store.dump())
            await first_started.wait()
            second = asyncio.create_task(store.dump())
            await asyncio.sleep(0)

            assert active_writers == 1
            assert max_active_writers == 1

            release_first.set()
            await asyncio.gather(first, second)
            assert write_count == 2
            assert max_active_writers == 1
            monkeypatch.setattr(store, "_write_sidecar", original_write_sidecar)
            await store.close()

    run(go())


def test_faiss_rebuild_skips_wrong_dimension_chunks():
    """FAISS rebuild should ignore chunks whose embedding dimensions do not match."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            try:
                store = FaissLocalFileStore(name="t_faiss_dim_filter", embedding_store="")
            except ImportError:
                pytest.skip("faiss is not installed")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store.file_chunks = {
                "good": chunk("good", "good.md", "alpha text"),
                "bad": chunk("bad", "bad.md", "alpha text"),
            }
            store.file_chunks["good"].embedding = np.array([1.0, 0.0], dtype=np.float16)
            store.file_chunks["bad"].embedding = np.array([1.0], dtype=np.float16)

            store._rebuild_index()

            assert set(store._id_to_row) == {"good"}
            assert store._faiss_index.ntotal == 1
            await store.close()

    run(go())


# -- Date filter tests -------------------------------------------------------


def test_date_filter_extract_and_match():
    """_extract_date_from_path and _matches_search_filter date filtering."""
    # Extract from various path formats
    assert LocalFileStore._extract_date_from_path("daily/2026-05-18/note.md") == "2026-05-18"
    assert LocalFileStore._extract_date_from_path("daily/2026-05-18.md") == "2026-05-18"
    assert LocalFileStore._extract_date_from_path("resource/2026-06-06/report.pdf") == "2026-06-06"
    assert LocalFileStore._extract_date_from_path("digest/personal/topic.md") is None
    assert LocalFileStore._extract_date_from_path("daily/9999-99-99/note.md") is None
    assert LocalFileStore._extract_date_from_path("note.md") is None

    # start_date / end_date boundary checks
    filt = {"start_date": "2026-02-01", "end_date": "2026-02-28"}
    assert LocalFileStore._matches_search_filter(chunk("a", "daily/2026-01-31/n.md", "t"), filt) is False
    assert LocalFileStore._matches_search_filter(chunk("b", "daily/2026-02-01/n.md", "t"), filt) is True
    assert LocalFileStore._matches_search_filter(chunk("c", "daily/2026-02-15/n.md", "t"), filt) is True
    assert LocalFileStore._matches_search_filter(chunk("d", "daily/2026-02-28/n.md", "t"), filt) is True
    assert LocalFileStore._matches_search_filter(chunk("e", "daily/2026-03-01/n.md", "t"), filt) is False

    # No date in path → not excluded (non-strict, default)
    assert LocalFileStore._matches_search_filter(chunk("x", "digest/personal/topic.md", "t"), filt) is True

    # strict_date_filter=True → no-date paths excluded when date filter is active
    strict_filt = {**filt, "strict_date_filter": True}
    assert LocalFileStore._matches_search_filter(chunk("x", "digest/personal/topic.md", "t"), strict_filt) is False
    assert LocalFileStore._matches_search_filter(chunk("b", "daily/2026-02-15/n.md", "t"), strict_filt) is True

    # strict_date_filter=True but no date bounds → no-date paths still pass
    strict_no_bounds = {"strict_date_filter": True}
    assert LocalFileStore._matches_search_filter(chunk("x", "digest/personal/topic.md", "t"), strict_no_bounds) is True

    # start_date/end_date stay in reserved, not leaked to metadata
    c = chunk("z", "daily/2026-05-18/note.md", "text")
    assert LocalFileStore._matches_search_filter(c, {"start_date": "2026-01-01", "end_date": "2026-12-31"}) is True


def test_date_filter_with_vector_and_keyword_search():
    """vector_search and keyword_search respect start_date/end_date filters."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = LocalFileStore(name="t_date_search", embedding_store="")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()

            await store.upsert(
                [
                    (node("daily/2026-01-10/a.md"), [chunk("a", "daily/2026-01-10/a.md", "alpha topic")]),
                    (node("daily/2026-02-15/b.md"), [chunk("b", "daily/2026-02-15/b.md", "alpha topic")]),
                    (node("daily/2026-03-20/c.md"), [chunk("c", "daily/2026-03-20/c.md", "alpha topic")]),
                ],
            )

            filt = {"start_date": "2026-02-01", "end_date": "2026-02-28"}
            assert [c.id for c in await store.vector_search("alpha", 5, filt)] == ["b"]
            assert [c.id for c in await store.keyword_search("alpha", 5, filt)] == ["b"]

            # start_date only
            assert sorted(c.id for c in await store.vector_search("alpha", 5, {"start_date": "2026-02-01"})) == [
                "b",
                "c",
            ]
            # end_date only
            assert sorted(c.id for c in await store.keyword_search("alpha", 5, {"end_date": "2026-02-28"})) == [
                "a",
                "b",
            ]

            await store.close()

    run(go())


def test_faiss_date_filter_progressive_recall():
    """FaissLocalFileStore progressive recall collects enough results with date filter."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            try:
                store = FaissLocalFileStore(name="t_faiss_date", embedding_store="")
            except ImportError:
                pytest.skip("faiss is not installed")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()

            files = []
            for i in range(10):
                day = f"2026-01-{i + 10:02d}"
                path = f"daily/{day}/note.md"
                files.append((node(path), [chunk(f"c{i}", path, "alpha topic")]))
            await store.upsert(files)

            # Enough matches exist
            results = await store.vector_search("alpha", 3, {"start_date": "2026-01-15", "end_date": "2026-01-17"})
            assert len(results) == 3

            # Fewer matches than limit → returns all matching
            results = await store.vector_search("alpha", 5, {"start_date": "2026-01-18", "end_date": "2026-01-19"})
            assert len(results) == 2

            await store.close()

    run(go())


def test_faiss_vector_search_survives_concurrent_index_drop():
    """vector_search must not dereference a None index if a concurrent clear()
    drops it while the query embedding is being computed.

    Reproduces the TOCTOU: the entry guard sees a live index, get_embedding
    yields control, and a concurrent clear() (after embedding was disabled)
    sets self._faiss_index to None. The post-await re-check must return [].
    """

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_faiss_store("t_faiss_search_drop")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()
            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])

            real_get_embedding = store.embedding_store.get_embedding

            async def dropping_get_embedding(text, **kwargs):
                emb = await real_get_embedding(text, **kwargs)
                # Simulate a concurrent clear() landing during the await.
                store._faiss_index = None
                return emb

            store.embedding_store.get_embedding = dropping_get_embedding

            # Must return [] rather than raising AttributeError on None.ntotal.
            assert await store.vector_search("alpha", 5, {}) == []
            await store.close()

    run(go())


# -- Async reindex tests -----------------------------------------------------


def _new_faiss_store(name, **kwargs):
    """Construct a started FAISS store with a fake embedding backend and empty index."""
    try:
        store = FaissLocalFileStore(name=name, embedding_store="", **kwargs)
    except ImportError:
        pytest.skip("faiss is not installed")
    return store


async def _settle_reindex(store, timeout=5.0):
    """Wait until no async reindex is pending or in flight."""
    deadline = time.monotonic() + timeout
    while store._reindex_event.is_set() or store._reindex_busy:
        if time.monotonic() > deadline:
            raise AssertionError("async reindex did not settle in time")
        await asyncio.sleep(0.005)


def test_faiss_async_reindex_disabled_by_default():
    """Default store keeps the synchronous compaction path: no background worker."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_faiss_store("t_faiss_sync_default", max_tombstones=2)
            assert store.async_reindex is False
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()

            files = [(node(f"n{i}.md"), [chunk(f"c{i}", f"n{i}.md", "alpha text")]) for i in range(4)]
            await store.upsert(files)
            await store.delete([f"n{i}.md" for i in range(3)])

            # Synchronous rebuild ran inline; no background worker was created.
            assert store._reindex_worker_task is None
            assert store._tombstones == set()
            assert set(store._id_to_row) == {"c3"}
            assert [c.id for c in await store.vector_search("alpha", 10, {})] == ["c3"]
            await store.close()

    run(go())


def test_faiss_async_reindex_triggered_by_compaction():
    """Crossing the tombstone threshold submits a background rebuild whose result
    matches a synchronous rebuild (deleted ids gone, tombstones cleared)."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_faiss_store("t_faiss_async_compact", async_reindex=True, max_tombstones=2)
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()

            files = [(node(f"n{i}.md"), [chunk(f"c{i}", f"n{i}.md", "alpha text")]) for i in range(4)]
            await store.upsert(files)
            assert store._reindex_worker_task is None  # below threshold, nothing submitted yet

            await store.delete([f"n{i}.md" for i in range(3)])  # 3 tombstones >= 2 -> submit
            assert store._reindex_worker_task is not None
            await _settle_reindex(store)

            assert set(store._id_to_row) == {"c3"}
            assert store._tombstones == set()
            assert [c.id for c in await store.vector_search("alpha", 10, {})] == ["c3"]
            await store.close()

    run(go())


def test_faiss_async_reindex_no_lost_writes_during_build():
    """Writes that land while an async rebuild is in flight are not lost: a
    follow-up rebuild folds them in (eventual consistency)."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_faiss_store("t_faiss_async_nolost", async_reindex=True)
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()

            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            assert [c.id for c in await store.vector_search("alpha", 5, {})] == ["a"]

            started = threading.Event()
            release = threading.Event()
            real_build = store._build_index_blocking

            def gated_build(dim, vectors):
                started.set()
                release.wait()
                return real_build(dim, vectors)

            store._build_index_blocking = gated_build

            store._submit_reindex()  # snapshot == {a}
            while not started.is_set():
                await asyncio.sleep(0.005)

            # Concurrent writes on the live index while the build is blocked:
            await store.upsert([(node("b.md"), [chunk("b", "b.md", "beta text")])])  # brand new
            await store.upsert([(node("a.md"), [chunk("a", "a.md", "beta text")])])  # changed text

            release.set()
            await _settle_reindex(store)

            # After the follow-up rebuild the index reflects both concurrent writes;
            # the changed chunk now embeds as "beta".
            assert set(store._id_to_row) == {"a", "b"}
            assert {c.id for c in await store.vector_search("beta", 5, {})} == {"a", "b"}
            await store.close()

    run(go())


def test_faiss_async_reindex_single_worker_coalesces():
    """Only one reindex runs at a time; repeated submissions coalesce and the
    worker stays a single long-lived task."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_faiss_store("t_faiss_async_single", async_reindex=True)
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()
            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])

            started = threading.Event()
            release = threading.Event()
            active = {"n": 0}
            peak = {"n": 0}
            real_build = store._build_index_blocking

            def gated_build(dim, vectors):
                active["n"] += 1
                peak["n"] = max(peak["n"], active["n"])
                started.set()
                release.wait()
                try:
                    return real_build(dim, vectors)
                finally:
                    active["n"] -= 1

            store._build_index_blocking = gated_build

            store._submit_reindex()
            while not started.is_set():
                await asyncio.sleep(0.005)
            worker = store._reindex_worker_task

            # Several more submissions while the first build is blocked collapse into
            # the flag rather than spawning parallel builds or a second worker.
            for _ in range(5):
                store._submit_reindex()
            assert store._reindex_worker_task is worker

            release.set()
            await _settle_reindex(store)

            assert peak["n"] == 1  # never two builds at once
            assert set(store._id_to_row) == {"a"}
            await store.close()

    run(go())


def test_faiss_async_reindex_cancelled_on_close():
    """close() stops an in-flight reindex without hanging on the worker thread."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_faiss_store("t_faiss_async_close", async_reindex=True)
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()
            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])

            started = threading.Event()

            def slow_build(_dim, _vectors):
                started.set()
                while not store._closing:
                    time.sleep(0.005)

            store._build_index_blocking = slow_build
            store._submit_reindex()
            while not started.is_set():
                await asyncio.sleep(0.005)

            await store.close()  # sets _closing, cancels the worker; the build thread exits
            assert store._reindex_worker_task is None

    run(go())


def test_faiss_close_does_not_leave_orphan_reindex():
    """The final dump in _close() must not submit a background reindex."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_faiss_store("t_faiss_close_orphan", async_reindex=True, max_tombstones=2)
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()

            files = [(node(f"n{i}.md"), [chunk(f"c{i}", f"n{i}.md", "alpha text")]) for i in range(4)]
            await store.upsert(files)

            # Block the build so the reindex is still in flight (tombstones uncompacted)
            # when close runs.
            started = threading.Event()

            def slow_build(_dim, _vectors):
                started.set()
                while not store._closing:
                    time.sleep(0.005)

            store._build_index_blocking = slow_build
            await store.delete([f"n{i}.md" for i in range(3)])  # 3 tombstones >= 2 -> submit
            while not started.is_set():
                await asyncio.sleep(0.005)
            assert len(store._tombstones) >= store.max_tombstones  # not compacted yet

            # Closing cancels the in-flight worker; the final dump -> _compact_if_needed
            # still sees tombstones over the threshold, but the _closing guard stops it
            # from submitting a new (orphan) reindex.
            await store.close()
            assert store._reindex_worker_task is None

    run(go())


def test_faiss_clear_waits_for_in_flight_dump():
    """clear() serializes with dump() through _faiss_dump_lock."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_faiss_store("t_faiss_clear_lock")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()
            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])

            first_started = asyncio.Event()
            release = asyncio.Event()
            original_write_sidecar = store._write_sidecar

            async def blocking_write_sidecar():
                first_started.set()
                await release.wait()

            store._write_sidecar = blocking_write_sidecar
            dump_task = asyncio.create_task(store.dump())
            await first_started.wait()

            clear_task = asyncio.create_task(store.clear())
            await asyncio.sleep(0.02)
            assert not clear_task.done()  # blocked on _faiss_dump_lock held by dump

            release.set()
            await asyncio.gather(dump_task, clear_task)
            assert store._id_to_row == {}

            store._write_sidecar = original_write_sidecar
            await store.close()

    run(go())


def test_faiss_delete_queries_graph_once():
    """delete() resolves nodes once and reuses them (no redundant get_nodes)."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_faiss_store("t_faiss_delete_once")
            await store.start()
            store.embedding_store = FakeEmbeddingStore()
            store._faiss_index = store._new_index()
            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])

            calls: list = []
            original_get_nodes = store.file_graph.get_nodes

            async def counting_get_nodes(paths=None):
                calls.append(paths)
                return await original_get_nodes(paths)

            store.file_graph.get_nodes = counting_get_nodes
            await store.delete("a.md")

            assert len(calls) == 1
            assert "a" not in store._id_to_row

            store.file_graph.get_nodes = original_get_nodes
            await store.close()

    run(go())
