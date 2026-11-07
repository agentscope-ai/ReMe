"""Keyword updates retain unchanged chunks without changing live BM25 scores."""

# pylint: disable=protected-access

from unittest.mock import AsyncMock, patch

import pytest

from reme.components.file_store import LocalFileStore
from reme.components.keyword_index import BM25Index

from .file_store_test_support import _new_faiss_store, _new_zvec_store, chunk, node


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["local", "faiss", "zvec"])
async def test_small_edit_only_tokenizes_changed_chunk(tmp_path, monkeypatch, backend):
    """A small edit adds one physical row and an identical update adds none."""
    monkeypatch.chdir(tmp_path)
    factories = {"local": LocalFileStore, "faiss": _new_faiss_store, "zvec": _new_zvec_store}
    kwargs = {"embedding_store": ""} if backend == "local" else {}
    store = factories[backend](name="delta", **kwargs)
    await store.start()
    try:
        await store.upsert(
            [(node("a.md"), [chunk("a", "a.md", "alpha memory"), chunk("b", "a.md", "beta memory")])],
        )
        index = store.keyword_index
        original_slot = index._doc_id_to_idx["a"]
        updated = [
            chunk("a", "a.md", "alpha memory", category="updated"),
            chunk("c", "a.md", "gamma memory"),
        ]
        with patch.object(index, "_tokenize", wraps=index._tokenize) as tokenize:
            await store.upsert([(node("a.md"), updated)])
            tokenize.assert_called_once_with("gamma memory")
        assert index._doc_id_to_idx["a"] == original_slot
        assert len(index._doc_ids) == 3
        assert set(index.document_ids) == {"a", "c"}
        assert store.file_chunks["a"].metadata == {"category": "updated"}

        with patch.object(index, "add_docs", wraps=index.add_docs) as add_docs:
            with patch.object(index, "delete_docs", wraps=index.delete_docs) as delete_docs:
                await store.upsert([(node("a.md"), updated)])
                add_docs.assert_not_called()
                delete_docs.assert_not_called()
        assert len(index._doc_ids) == 3
        assert await store.keyword_search("beta", 10, {}) == []
        assert [result.id for result in await store.keyword_search("gamma", 10, {})] == ["c"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_delta_scores_match_full_replacement_and_survive_reload(tmp_path, monkeypatch):
    """Cover same-ID edits, empty text, no tokens, shifted IDs and file removal."""
    monkeypatch.chdir(tmp_path)
    store = LocalFileStore(name="delta", embedding_store="")
    reference = BM25Index(name="reference")
    await store.start()
    await reference.start()
    revisions = [
        [("a", "alpha beta"), ("b", "beta gamma"), ("c", "gamma gamma")],
        [("a", "alpha beta"), ("b", "beta delta"), ("d", "gamma delta")],
        [("a", ""), ("b", "!!!"), ("d", "gamma delta")],
        [("a", "alpha beta"), ("b", "beta gamma"), ("d", "gamma delta")],
        [("shift-a", "alpha beta"), ("shift-b", "beta gamma"), ("shift-d", "gamma delta")],
        [],
    ]
    old_ids = []
    try:
        await store.upsert([(node("other.md"), [chunk("other", "other.md", "stable alpha")])])
        await reference.add_docs({"other": "stable alpha"})
        for revision in revisions:
            await store.upsert([(node("a.md"), [chunk(cid, "a.md", text) for cid, text in revision])])
            await reference.delete_docs(old_ids)
            await reference.add_docs({cid: text for cid, text in revision if text})
            old_ids = [cid for cid, _ in revision]
            assert set(store.keyword_index.document_ids) == set(reference.document_ids)
            assert store.keyword_index.total_len == reference.total_len
            for query in ("alpha", "beta", "gamma delta", "stable"):
                scores = await store.keyword_index.retrieve(query, limit=100)
                expected = await reference.retrieve(query, limit=100)
                assert scores == pytest.approx(expected, abs=1e-6)
            assert (await store.get_nodes(["a.md"]))[0].chunk_ids == old_ids

        await store.upsert([(node("a.md"), [chunk("final", "a.md", "delta memory")])])
        await store.dump()
    finally:
        await reference.close()
        await store.close()

    restored = LocalFileStore(name="delta", embedding_store="")
    await restored.start()
    try:
        assert set(restored.keyword_index.document_ids) == {"other", "final"}
        assert [result.id for result in await restored.keyword_search("delta", 10, {})] == ["final"]
        await restored.delete("a.md")
        assert set(restored.keyword_index.document_ids) == {"other"}
        assert await restored.keyword_search("delta", 10, {}) == []
    finally:
        await restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["keyword", "graph"])
async def test_retry_after_failed_upsert_restores_keyword_document(tmp_path, monkeypatch, failure_stage):
    """Retry staged chunks whether the keyword entry is missing or still stale."""
    monkeypatch.chdir(tmp_path)
    store = LocalFileStore(name="retry", embedding_store="")
    await store.start()
    try:
        await store.upsert([(node("a.md"), [chunk("a", "a.md", "original memory")])])
        updated = [(node("a.md"), [chunk("a", "a.md", "replacement memory")])]
        target, method = (
            (store.keyword_index, "add_docs") if failure_stage == "keyword" else (store.file_graph, "upsert_nodes")
        )
        with patch.object(target, method, side_effect=RuntimeError("update failed")):
            with pytest.raises(RuntimeError, match="update failed"):
                await store.upsert(updated)
        assert bool(await store.keyword_search("original", 10, {})) == (failure_stage == "graph")
        await store.upsert(updated)
        assert [result.id for result in await store.keyword_search("replacement", 10, {})] == ["a"]
        assert await store.keyword_search("original", 10, {}) == []
        assert not store._keyword_pending_paths
    finally:
        await store.close()


class LegacyKeywordIndex:
    """A backend without live-ID introspection keeps receiving full replacements."""

    def __init__(self):
        self.add_docs = AsyncMock()
        self.delete_docs = AsyncMock()

    @property
    def document_ids(self):
        """Signal that live document IDs are unavailable."""
        raise NotImplementedError


@pytest.mark.asyncio
async def test_backend_without_live_ids_keeps_full_replacement(tmp_path, monkeypatch):
    """Preserve the contract for keyword backends that cannot report live IDs."""
    monkeypatch.chdir(tmp_path)
    store = LocalFileStore(name="legacy", embedding_store="")
    await store.start()
    original_index = store.keyword_index
    try:
        files = [(node("a.md"), [chunk("a", "a.md", "alpha memory")])]
        await store.upsert(files)
        legacy = LegacyKeywordIndex()
        store.keyword_index = legacy
        await store.upsert(files)
        legacy.delete_docs.assert_awaited_once_with(["a"])
        legacy.add_docs.assert_awaited_once_with({"a": "alpha memory"})
    finally:
        store.keyword_index = original_index
        await store.close()
