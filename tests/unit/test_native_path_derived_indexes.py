"""Regressions for canonical write batches and the derived indexes they feed.

The file graph merges records whose paths are one file spelled two ways. The
surviving record owns the only chunk handles the derived indexes may keep, and
every backend must consume the same canonical batch the store publishes.
"""

# pylint: disable=protected-access,import-outside-toplevel

import pickle
import tempfile
from contextlib import chdir as temp_chdir

import pytest

from reme.components.file_graph import LocalFileGraph
from reme.components.file_store import FaissLocalFileStore, LocalFileStore, ZvecLocalFileStore
from reme.schema import FileNode
from reme.utils.jsonl_zst import write_jsonl_zst

from .file_store_test_support import FakeEmbeddingStore, _new_faiss_store, _new_zvec_store, chunk, node


def _attach_vector_backend(store) -> None:
    """Give a store started without a provider a deterministic vector backend."""
    store.embedding_store = FakeEmbeddingStore()
    if isinstance(store, FaissLocalFileStore):
        store._faiss_index = store._new_index()
    elif isinstance(store, ZvecLocalFileStore) and store._collection is None:
        store._collection = store._create_collection()


@pytest.mark.parametrize("store_factory", [_new_faiss_store, _new_zvec_store])
@pytest.mark.asyncio
async def test_native_path_insert_reaches_vector_index(store_factory):
    """A file first written through native separators is searchable by vector."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = store_factory("t_native_insert_vector")
        await store.start()
        try:
            _attach_vector_backend(store)
            await store.upsert([(node(r"daily\a.md"), [chunk("a", r"daily\a.md", "alpha text")])])

            assert [file_node.path for file_node in await store.get_nodes()] == ["daily/a.md"]
            assert [hit.id for hit in await store.vector_search("alpha", 5, {})] == ["a"]
        finally:
            await store.close()


@pytest.mark.parametrize("store_factory", [_new_faiss_store, _new_zvec_store])
@pytest.mark.asyncio
async def test_native_path_update_replaces_previous_vector(store_factory):
    """Rewriting one file through native separators drops the old vector."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = store_factory("t_native_update_vector")
        await store.start()
        try:
            _attach_vector_backend(store)
            await store.upsert([(node("daily/a.md"), [chunk("old", "daily/a.md", "alpha old")])])
            await store.upsert([(node(r"daily\a.md"), [chunk("new", r"daily\a.md", "fresh rewritten")])])

            assert set(store.file_chunks) == {"new"}
            # "alpha" embeds to the retired chunk's vector, so a stale row would
            # use up the single slot and then be discarded, leaving no result.
            assert [hit.id for hit in await store.vector_search("alpha", 1, {})] == ["new"]
            assert [hit.id for hit in await store.vector_search("fresh", 1, {})] == ["new"]
        finally:
            await store.close()


def _record(path: str, mtime: float, chunk_ids: list[str]) -> FileNode:
    """Build one persisted graph record with an explicit revision order."""
    return FileNode(path=path, st_mtime=mtime, chunk_ids=chunk_ids)


def _legacy_records() -> list[FileNode]:
    """Three spellings of one file, the last of which reuses the first handle."""
    return [
        _record(r"daily\sub/a.md", 1.0, ["shared", "old"]),
        _record(r"daily/sub\a.md", 2.0, ["middle"]),
        _record("daily/sub/a.md", 3.0, ["shared", "current"]),
    ]


def _seed_local_graph(graph: LocalFileGraph) -> None:
    graph._graph_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_zst(graph._graph_file, [record.model_dump_json() for record in _legacy_records()])


def _seed_nx_graph(graph) -> None:
    pytest.importorskip("networkx")
    import networkx as nx

    persisted = nx.MultiDiGraph()
    for record in _legacy_records():
        persisted.add_node(record.path, node=record)
    graph._graph_file.parent.mkdir(parents=True, exist_ok=True)
    graph._graph_file.write_bytes(pickle.dumps(persisted))


@pytest.mark.parametrize("backend", ["local", "nx"])
@pytest.mark.asyncio
async def test_merge_reports_only_handles_no_surviving_record_owns(backend, monkeypatch, tmp_path):
    """A handle reused by the winning record is not a superseded handle."""
    monkeypatch.chdir(tmp_path)
    if backend == "local":
        graph = LocalFileGraph(name="merge_probe")
        _seed_local_graph(graph)
    else:
        pytest.importorskip("networkx")
        from reme.components.file_graph import NxFileGraph

        graph = NxFileGraph(name="merge_probe")
        _seed_nx_graph(graph)

    await graph.load()
    try:
        nodes = await graph.get_nodes()
        assert [file_node.path for file_node in nodes] == ["daily/sub/a.md"]
        assert list(nodes[0].chunk_ids) == ["shared", "current"]
        assert await graph.take_dropped_chunk_ids() == ["middle", "old"]
    finally:
        await graph.close()


@pytest.mark.asyncio
async def test_startup_keeps_chunks_the_merged_record_still_owns(monkeypatch, tmp_path):
    """Startup cleanup must not make the consistency check clear the index."""
    monkeypatch.chdir(tmp_path)
    # Standalone mode resolves ``file_graph="default"`` into a default-named
    # backend, so seeding that instance writes the file the store will load.
    _seed_local_graph(LocalFileGraph())

    store = LocalFileStore(name="t_legacy_merge", embedding_store="")
    store.chunks_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_zst(
        store.chunks_path,
        [
            store._serialize_chunk(chunk("shared", "daily/sub/a.md", "shared body")),
            store._serialize_chunk(chunk("old", "daily/sub/a.md", "superseded old body")),
            store._serialize_chunk(chunk("middle", "daily/sub/a.md", "middle body")),
            store._serialize_chunk(chunk("current", "daily/sub/a.md", "current body")),
        ],
    )

    await store.start()
    try:
        assert set(store.file_chunks) == {"shared", "current"}
        nodes = await store.get_nodes()
        assert [list(file_node.chunk_ids) for file_node in nodes] == [["shared", "current"]]
        assert [hit.id for hit in await store.keyword_search("superseded", 5, {})] == []
    finally:
        await store.close()
