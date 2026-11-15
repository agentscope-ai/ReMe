"""Tests for FAISS index maintenance: backfill incremental sync and idle-time optimize_index."""

# pylint: disable=protected-access

from contextlib import chdir as temp_chdir
import asyncio
import tempfile

import pytest


from reme.components.file_store import LocalFileStore
from reme.steps.index import OptimizeIndexStep

from .file_store_test_support import (
    FakeEmbeddingStore,
    _new_faiss_store,
    _seed_unembedded_chunk,
    _settle_reindex,
    chunk,
    node,
)


@pytest.mark.asyncio
async def test_faiss_embedding_reindex_and_close_complete_concurrently(tmp_path, monkeypatch):
    """Shutdown must not take the FAISS lock while reindex holds maintenance."""
    monkeypatch.chdir(tmp_path)
    embedding_started = asyncio.Event()
    release_embedding = asyncio.Event()
    shutdown_checkpoint = asyncio.Event()

    class BlockingEmbeddingStore(FakeEmbeddingStore):
        """Pause a real embedding reindex at its provider boundary."""

        async def get_node_embeddings(self, nodes, **kwargs):
            embedding_started.set()
            await release_embedding.wait()
            return await super().get_node_embeddings(nodes, **kwargs)

    store = _new_faiss_store("t_reindex_close")
    await store.start()
    tasks = []
    close_task = None
    try:
        store.embedding_store = FakeEmbeddingStore()
        store._faiss_index = store._new_index()
        await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha memory")])])
        store.embedding_store = BlockingEmbeddingStore()
        dump_owned_state = store._dump_owned_state

        async def observe_shutdown_checkpoint():
            if asyncio.current_task() is close_task:
                shutdown_checkpoint.set()
            await dump_owned_state()

        monkeypatch.setattr(store, "_dump_owned_state", observe_shutdown_checkpoint)
        reindex_task = asyncio.create_task(store.reindex("embedding"))
        tasks.append(reindex_task)
        await asyncio.wait_for(embedding_started.wait(), timeout=3)
        close_task = asyncio.create_task(store.close())
        tasks.append(close_task)
        await asyncio.wait_for(shutdown_checkpoint.wait(), timeout=3)
        release_embedding.set()
        result, _ = await asyncio.wait_for(asyncio.gather(*tasks), timeout=3)
        assert result == {"indexed": 1, "scope": "embedding"}
        assert store.faiss_path.exists()
        assert store.faiss_idmap_path.exists()
    finally:
        release_embedding.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await store.close()

    restored = _new_faiss_store("t_reindex_close")
    restored.embedding_store = FakeEmbeddingStore()
    await restored.start()
    try:
        assert restored._faiss_index.ntotal == 1
        assert [result.id for result in await restored.vector_search("alpha", 5, {})] == ["a"]
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_faiss_backfill_adds_incrementally_without_rebuild():
    """Backfilled vectors are added to the live index; no full rebuild happens
    while tombstones stay under the compaction threshold."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = _new_faiss_store("t_faiss_backfill_incr")
        await store.start()
        store.embedding_store = FakeEmbeddingStore()
        store._faiss_index = store._new_index()

        await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
        await store.upsert([(node("b.md"), [chunk("b", "b.md", "beta text")])])
        # Same-id text change tombstones the old row (below the 128 floor).
        await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha updated")])])
        assert store._tombstones == {0}

        await _seed_unembedded_chunk(store, "c", "c.md", "alpha extra")
        await store._backfill_missing_embeddings()

        # The new vector is live; the surviving tombstone proves the index
        # was extended in place rather than rebuilt.
        assert set(store._id_to_row) == {"a", "b", "c"}
        assert store._tombstones == {0}
        assert store._reindex_worker_task is None
        assert {c.id for c in await store.vector_search("alpha", 10, {})} == {"a", "b", "c"}
        await store.close()


@pytest.mark.asyncio
async def test_faiss_backfill_compacts_when_tombstones_cross_threshold():
    """Tombstone pressure at backfill time still triggers a full rebuild."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = _new_faiss_store("t_faiss_backfill_compact")
        await store.start()
        store.embedding_store = FakeEmbeddingStore()
        store._faiss_index = store._new_index()

        await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
        await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha updated")])])
        assert store._tombstones == {0}

        # Lower the threshold only now, so the upsert above did not compact.
        store.max_tombstones = 1
        await _seed_unembedded_chunk(store, "c", "c.md", "alpha extra")
        await store._backfill_missing_embeddings()

        assert store._tombstones == set()
        assert set(store._id_to_row) == {"a", "c"}
        assert {c.id for c in await store.vector_search("alpha", 10, {})} == {"a", "c"}
        await store.close()


@pytest.mark.asyncio
async def test_faiss_backfill_mass_delta_uses_async_reindex():
    """With async_reindex, a delta that rivals the live rows (initial mass
    backfill into an empty index) goes through the background worker."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = _new_faiss_store("t_faiss_backfill_async_mass", async_reindex=True)
        await store.start()
        store.embedding_store = FakeEmbeddingStore()
        store._faiss_index = store._new_index()

        await _seed_unembedded_chunk(store, "a", "a.md", "alpha text")
        await _seed_unembedded_chunk(store, "b", "b.md", "beta text")
        await store._backfill_missing_embeddings()

        assert store._reindex_worker_task is not None  # routed off-loop
        await _settle_reindex(store)
        assert set(store._id_to_row) == {"a", "b"}
        assert [c.id for c in await store.vector_search("alpha", 5, {})][0] == "a"
        await store.close()


@pytest.mark.asyncio
async def test_faiss_backfill_small_delta_adds_inline_in_async_mode():
    """With async_reindex, a small delta is added inline without a worker."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = _new_faiss_store("t_faiss_backfill_async_small", async_reindex=True)
        await store.start()
        store.embedding_store = FakeEmbeddingStore()
        store._faiss_index = store._new_index()

        await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
        await store.upsert([(node("b.md"), [chunk("b", "b.md", "beta text")])])

        await _seed_unembedded_chunk(store, "c", "c.md", "alpha extra")
        await store._backfill_missing_embeddings()

        # 1 new vector < 2 live rows -> inline add, no background worker.
        assert store._reindex_worker_task is None
        assert set(store._id_to_row) == {"a", "b", "c"}
        assert {c.id for c in await store.vector_search("alpha", 10, {})} >= {"a", "c"}
        await store.close()


# -- optimize_index (idle-time maintenance) -------------------------------------


@pytest.mark.asyncio
async def test_tombstone_threshold_scales_and_honors_override():
    """Threshold math: full vs half scale, 128 floor, and max_tombstones override."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = _new_faiss_store("t_faiss_threshold")
        await store.start()
        store.embedding_store = FakeEmbeddingStore()
        store._faiss_index = type("FakeIndex", (), {"ntotal": 1000})()

        assert store._tombstone_threshold() == 300  # 0.3 * 1000
        assert store._tombstone_threshold(scale=0.5) == 150  # half the write-path bar
        store._faiss_index = type("FakeIndex", (), {"ntotal": 100})()
        assert store._tombstone_threshold(scale=0.5) == 128  # floor dominates

        store.max_tombstones = 7
        assert store._tombstone_threshold() == 7
        assert store._tombstone_threshold(scale=0.5) == 7  # fixed override ignores scale

        store._faiss_index = None  # avoid persisting the fake index on close
        await store.close()


@pytest.mark.asyncio
async def test_local_store_optimize_index_is_noop():
    """LocalFileStore.optimize_index() completes without touching store state."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = LocalFileStore(name="t_local_optimize", embedding_store="")
        await store.start()
        await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])

        await store.optimize_index()

        assert set(store.file_chunks) == {"a"}
        assert [c.id for c in await store.keyword_search("alpha", 5, {})] == ["a"]
        await store.close()


@pytest.mark.asyncio
async def test_faiss_optimize_index_noop_below_threshold():
    """optimize_index() keeps tombstones when they are under the half bar."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = _new_faiss_store("t_faiss_optimize_noop")
        await store.start()
        store.embedding_store = FakeEmbeddingStore()
        store._faiss_index = store._new_index()

        await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha text")])])
        await store.upsert([(node("a.md"), [chunk("a", "a.md", "alpha updated")])])
        assert store._tombstones == {0}

        await store.optimize_index()  # 1 tombstone <= 128 floor -> untouched

        assert store._tombstones == {0}
        assert store._reindex_worker_task is None
        await store.close()


@pytest.mark.asyncio
async def test_faiss_optimize_index_compacts_inline_when_sync():
    """optimize_index() rebuilds inline once tombstones exceed the (overridden) bar."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = _new_faiss_store("t_faiss_optimize_sync")
        await store.start()
        store.embedding_store = FakeEmbeddingStore()
        store._faiss_index = store._new_index()

        files = [(node(f"n{i}.md"), [chunk(f"c{i}", f"n{i}.md", "alpha text")]) for i in range(4)]
        await store.upsert(files)
        updated = [(node(f"n{i}.md"), [chunk(f"c{i}", f"n{i}.md", f"alpha v2 {i}")]) for i in range(3)]
        await store.upsert(updated)
        assert len(store._tombstones) == 3

        store.max_tombstones = 2  # lower the bar only for optimize_index
        await store.optimize_index()

        assert store._tombstones == set()
        assert set(store._id_to_row) == {"c0", "c1", "c2", "c3"}
        assert store._reindex_worker_task is None  # inline path
        assert len(await store.vector_search("alpha", 10, {})) == 4
        await store.close()


@pytest.mark.asyncio
async def test_faiss_optimize_index_uses_worker_when_async():
    """optimize_index() submits the rebuild to the background worker under async_reindex."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = _new_faiss_store("t_faiss_optimize_async", async_reindex=True)
        await store.start()
        store.embedding_store = FakeEmbeddingStore()
        store._faiss_index = store._new_index()

        files = [(node(f"n{i}.md"), [chunk(f"c{i}", f"n{i}.md", "alpha text")]) for i in range(4)]
        await store.upsert(files)
        updated = [(node(f"n{i}.md"), [chunk(f"c{i}", f"n{i}.md", f"alpha v2 {i}")]) for i in range(3)]
        await store.upsert(updated)
        assert len(store._tombstones) == 3

        store.max_tombstones = 2
        await store.optimize_index()

        assert store._reindex_worker_task is not None  # routed off-loop
        await _settle_reindex(store)
        assert store._tombstones == set()
        assert set(store._id_to_row) == {"c0", "c1", "c2", "c3"}
        await store.close()


@pytest.mark.asyncio
async def test_optimize_index_step_calls_file_store_optimize_index():
    """The cron-facing step delegates to file_store.optimize_index() and reports success."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = LocalFileStore(name="t_optimize_step", embedding_store="")
        await store.start()

        calls = []
        original_optimize = store.optimize_index

        async def counting_optimize():
            calls.append(True)
            await original_optimize()

        store.optimize_index = counting_optimize
        step = OptimizeIndexStep(file_store=store)
        await step()

        assert calls == [True]
        assert step.context.response.metadata["optimized_index"] is True
        await store.close()
