"""Tests for ZvecLocalFileStore: native upsert/delete sync, persistence, and rebuild triggers."""

# pylint: disable=protected-access

import asyncio
import json
import os
import tempfile

import numpy as np
import pytest

from reme.components.file_store import ZvecLocalFileStore
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


def run(coro):
    """Run an async test body."""
    return asyncio.run(coro)


def node(path: str) -> FileNode:
    """Build a minimal file node."""
    return FileNode(path=path, st_mtime=1.0)


def chunk(chunk_id: str, path: str, text: str, **metadata) -> FileChunk:
    """Build a minimal file chunk."""
    return FileChunk(id=chunk_id, path=path, text=text, start_line=1, end_line=1, metadata=metadata)


def _new_zvec_store(name, **kwargs):
    """Construct a zvec store with embedding disabled at bind time."""
    try:
        store = ZvecLocalFileStore(name=name, embedding_store="", **kwargs)
    except ImportError:
        pytest.skip("zvec is not installed")
    return store


async def _started_store(name, **kwargs) -> ZvecLocalFileStore:
    """Start a store with a fake embedding provider and a live collection."""
    store = _new_zvec_store(name, **kwargs)
    store.embedding_store = FakeEmbeddingStore()
    await store.start()
    if store._collection is None:
        store._collection = store._create_collection()
    return store


async def _seed_unembedded_chunk(store: ZvecLocalFileStore, chunk_id: str, path: str, text: str) -> None:
    """Attach one chunk without a vector, keeping the graph invariant intact."""
    store.file_chunks[chunk_id] = chunk(chunk_id, path, text)
    file_node = node(path)
    file_node.chunk_ids = [chunk_id]
    await store.file_graph.upsert_nodes([file_node])


# -- CRUD / search ------------------------------------------------------------


def test_zvec_upsert_and_vector_search_ranks_by_similarity():
    """Upserted chunks are searchable; results are ranked by cosine similarity."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _started_store("t_zvec_basic")

            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            await store.upsert([(node("b.md"), [chunk("b", "b.md", "beta text")])])

            results = await store.vector_search("alpha", 10, {})
            assert [c.id for c in results] == ["a", "b"]
            # Identical vector -> cosine distance 0 -> similarity score 1.
            assert results[0].scores["vector"] == pytest.approx(1.0, abs=1e-5)
            assert results[0].scores["score"] == results[0].scores["vector"]
            await store.close()

    run(go())


def test_zvec_same_id_text_change_updates_vector_in_place():
    """A same-id text change replaces the vector via native zvec upsert."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _started_store("t_zvec_update")

            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            assert [c.id for c in await store.vector_search("alpha", 5, {})] == ["a"]

            await store.upsert([(node("a.md"), [chunk("a", "a.md", "beta text")])])

            assert store._indexed_ids == {"a"}
            results = await store.vector_search("beta", 5, {})
            assert [c.id for c in results] == ["a"]
            assert results[0].scores["vector"] == pytest.approx(1.0, abs=1e-5)
            await store.close()

    run(go())


def test_zvec_upsert_removes_stale_chunks_from_collection():
    """Re-upserting a path deletes vectors of chunks that no longer exist."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _started_store("t_zvec_stale")

            await store.upsert([(node("a.md"), [chunk("old", "a.md", "alpha old")])])
            await store.upsert([(node("a.md"), [chunk("new", "a.md", "alpha new")])])

            assert store._indexed_ids == {"new"}
            assert [c.id for c in await store.vector_search("alpha", 10, {})] == ["new"]
            await store.close()

    run(go())


def test_zvec_delete_removes_vectors():
    """Deleting a path removes its chunk vectors from the collection."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _started_store("t_zvec_delete")

            await store.upsert(
                [
                    (node("a.md"), [chunk("a", "a.md", "alpha text")]),
                    (node("b.md"), [chunk("b", "b.md", "beta text")]),
                ],
            )
            await store.delete("a.md")

            assert store._indexed_ids == {"b"}
            assert [c.id for c in await store.vector_search("alpha", 10, {})] == ["b"]
            await store.close()

    run(go())


def test_zvec_vector_search_applies_post_filter():
    """The shared search_filter semantics apply on top of ANN results."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _started_store("t_zvec_filter")

            await store.upsert(
                [
                    (node("a.md"), [chunk("a", "a.md", "alpha one", kind="x")]),
                    (node("b.md"), [chunk("b", "b.md", "alpha two", kind="y")]),
                ],
            )

            assert [c.id for c in await store.vector_search("alpha", 10, {"path": "b.md"})] == ["b"]
            assert [c.id for c in await store.vector_search("alpha", 10, {"kind": "x"})] == ["a"]
            assert await store.vector_search("alpha", 10, {"path": "c.md"}) == []
            await store.close()

    run(go())


def test_zvec_clear_resets_collection():
    """clear() drops the collection, sidecar, and indexed-id state."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _started_store("t_zvec_clear")

            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            await store.dump()
            assert store.zvec_sidecar_path.exists()

            await store.clear()

            assert store._indexed_ids == set()
            assert not store.zvec_sidecar_path.exists()
            assert await store.vector_search("alpha", 10, {}) == []
            # The store stays usable after clear.
            await store.upsert([(node("b.md"), [chunk("b", "b.md", "beta text")])])
            assert [c.id for c in await store.vector_search("beta", 10, {})] == ["b"]
            await store.close()

    run(go())


def test_zvec_keyword_only_mode_keeps_keyword_search_working():
    """Without an embedding store, vector search is empty but keyword search works."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = _new_zvec_store("t_zvec_keyword_only")
            await store.start()

            await store.upsert([(node("a.md"), [chunk("a", "a.md", "uniquezvecword only")])])

            assert store._collection is None
            assert await store.vector_search("uniquezvecword", 5, {}) == []
            assert [c.id for c in await store.keyword_search("uniquezvecword", 5, {})] == ["a"]
            await store.close()

    run(go())


# -- embedding backfill ---------------------------------------------------------


def test_zvec_backfill_adds_incrementally():
    """Backfilled vectors are upserted into the live collection without a rebuild."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _started_store("t_zvec_backfill")

            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            await _seed_unembedded_chunk(store, "c", "c.md", "alpha extra")

            rebuilds = []
            original_rebuild = store._rebuild_collection
            store._rebuild_collection = lambda: rebuilds.append(True) or original_rebuild()

            await store._backfill_missing_embeddings()

            assert not rebuilds
            assert store._indexed_ids == {"a", "c"}
            assert {c.id for c in await store.vector_search("alpha", 10, {})} == {"a", "c"}
            await store.close()

    run(go())


# -- persistence ----------------------------------------------------------------


def test_zvec_persistence_round_trip_reopens_without_rebuild():
    """dump() + a fresh store reattaches the persisted collection via the sidecar."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            seed = await _started_store("t_zvec_persist")
            await seed.upsert(
                [
                    (node("a.md"), [chunk("a", "a.md", "alpha text")]),
                    (node("b.md"), [chunk("b", "b.md", "beta text")]),
                ],
            )
            await seed.close()

            store = _new_zvec_store("t_zvec_persist")
            store.embedding_store = FakeEmbeddingStore()

            rebuilds = []
            original_rebuild = store._rebuild_collection
            store._rebuild_collection = lambda: rebuilds.append(True) or original_rebuild()

            await store.start()

            assert not rebuilds
            assert store._indexed_ids == {"a", "b"}
            assert [c.id for c in await store.vector_search("alpha", 5, {})][0] == "a"
            assert [c.id for c in await store.vector_search("beta", 5, {})][0] == "b"
            await store.close()

    run(go())


def test_zvec_digest_mismatch_triggers_rebuild():
    """A sidecar whose digest fell behind the chunk JSONL forces a clean rebuild."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            seed = await _started_store("t_zvec_digest")
            await seed.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            await seed.close()

            sidecar = json.loads(seed.zvec_sidecar_path.read_text())
            sidecar["digest"] = "0" * 64
            seed.zvec_sidecar_path.write_text(json.dumps(sidecar))

            store = _new_zvec_store("t_zvec_digest")
            store.embedding_store = FakeEmbeddingStore()

            rebuilds = []
            original_rebuild = store._rebuild_collection
            store._rebuild_collection = lambda: rebuilds.append(True) or original_rebuild()

            await store.start()

            assert rebuilds == [True]
            assert store._indexed_ids == {"a"}
            assert [c.id for c in await store.vector_search("alpha", 5, {})] == ["a"]
            await store.close()

    run(go())


def test_zvec_missing_sidecar_triggers_rebuild():
    """A collection directory without its sidecar cannot be trusted and is rebuilt."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            seed = await _started_store("t_zvec_no_sidecar")
            await seed.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            await seed.close()

            seed.zvec_sidecar_path.unlink()

            store = _new_zvec_store("t_zvec_no_sidecar")
            store.embedding_store = FakeEmbeddingStore()
            await store.start()

            assert store._indexed_ids == {"a"}
            assert [c.id for c in await store.vector_search("alpha", 5, {})] == ["a"]
            await store.close()

    run(go())


def test_zvec_hnsw_m_mismatch_triggers_rebuild():
    """A persisted collection built with a different HNSW M is rebuilt."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            seed = await _started_store("t_zvec_m_mismatch", hnsw_m=16)
            await seed.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            await seed.close()

            store = _new_zvec_store("t_zvec_m_mismatch", hnsw_m=48)
            store.embedding_store = FakeEmbeddingStore()

            rebuilds = []
            original_rebuild = store._rebuild_collection
            store._rebuild_collection = lambda: rebuilds.append(True) or original_rebuild()

            await store.start()

            assert rebuilds == [True]
            assert [c.id for c in await store.vector_search("alpha", 5, {})] == ["a"]
            await store.close()

    run(go())


def test_zvec_stale_embedding_dimension_rebuilds_from_backfill():
    """Persisted vectors with a stale dimension are dropped and re-embedded."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            seed = await _started_store("t_zvec_stale_dim")
            await seed.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            await seed.close()

            class WideEmbeddingStore(FakeEmbeddingStore):
                """Same provider with a different dimension."""

                dimensions = 4

                def _embed(self, text: str) -> np.ndarray:
                    base = super()._embed(text)
                    return np.concatenate([base, base]).astype(np.float16)

            store = _new_zvec_store("t_zvec_stale_dim")
            store.embedding_store = WideEmbeddingStore()
            await store.start()
            # Startup backfill re-embeds the stale chunk in the background.
            await store._embedding_backfill_task

            assert store._collection.schema.vectors[0].dimension == 4
            assert store._indexed_ids == {"a"}
            assert [c.id for c in await store.vector_search("alpha", 5, {})] == ["a"]
            await store.close()

    run(go())


# -- maintenance ------------------------------------------------------------


def test_zvec_optimize_index_runs_native_optimize():
    """optimize_index() delegates to Collection.optimize() and keeps search intact."""

    async def go():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _started_store("t_zvec_optimize")

            await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
            await store.optimize_index()

            assert [c.id for c in await store.vector_search("alpha", 5, {})] == ["a"]
            await store.close()

    run(go())
