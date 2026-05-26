"""Tests for LocalFileStore and FaissLocalFileStore."""

# pylint: disable=protected-access

import asyncio
import hashlib
import importlib.util
import os
import tempfile
import warnings

import numpy as np

from reme4.components.embedding import BaseEmbeddingModel
from reme4.components.file_store import FaissLocalFileStore, LocalFileStore
from reme4.schema import FileChunk, FileNode

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

_FAISS_AVAILABLE = importlib.util.find_spec("faiss") is not None


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Deterministic stub: each lowercased word adds 1.0 at hash(word) % dim."""

    def __init__(self, dimensions: int = 8, **kwargs):
        super().__init__(model_name="fake", dimensions=dimensions, enable_cache=False, **kwargs)

    async def _get_embeddings(self, input_text, **kwargs):
        out = []
        for t in input_text:
            v = np.zeros(self.dimensions, dtype=np.float32)
            for w in t.lower().split():
                idx = int.from_bytes(hashlib.md5(w.encode()).digest()[:2], "big") % self.dimensions
                v[idx] += 1.0
            out.append(v.tolist())
        return out

    async def health_check(self, timeout: float = 2.0) -> bool:
        self.is_healthy = True
        return True


class temp_chdir:
    """Context manager to temporarily chdir into a path and restore on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


async def make_store(store_name: str = "test_store", **kwargs) -> LocalFileStore:
    """Build a started LocalFileStore with embedding disabled (no OpenAI dep)."""
    store = LocalFileStore(store_name=store_name, embedding_model="", **kwargs)
    await store.start()
    return store


def make_file(
    path: str,
    text: str,
    chunk_count: int = 1,
) -> tuple[FileNode, list[FileChunk]]:
    """Build a (FileNode, [FileChunk]) tuple ready for upsert_file."""
    chunks = [
        FileChunk(id=f"{path}::chunk{i}", path=path, text=f"{text} part{i}", start_line=i, end_line=i + 1)
        for i in range(chunk_count)
    ]
    node = FileNode(path=path, st_mtime=1.0, chunk_ids=[c.id for c in chunks])
    return node, chunks


def test_upsert_single_file():
    """upsert_file with a one-element list stores chunks and node."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()

            node, chunks = make_file("a.md", "hello world", chunk_count=2)
            await store.upsert([(node, chunks)])

            # Chunks landed in memory
            assert len(store.file_chunks) == 2
            assert {c.path for c in store.file_chunks.values()} == {"a.md"}
            # Node landed in graph
            nodes = await store.get_nodes(["a.md"])
            assert len(nodes) == 1
            assert sorted(nodes[0].chunk_ids) == sorted([c.id for c in chunks])

            await store.close()
            print("✓ test_upsert_single_file passed")

    asyncio.run(run())


def test_upsert_multiple_files():
    """upsert_file accepts a list of tuples and indexes them all."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()

            files = [make_file("a.md", "alpha"), make_file("b.md", "beta")]
            await store.upsert(files)

            assert len(store.file_chunks) == 2
            paths = {n.path for n in await store.get_nodes()}
            assert paths == {"a.md", "b.md"}

            await store.close()
            print("✓ test_upsert_multiple_files passed")

    asyncio.run(run())


def test_upsert_replaces_old_chunks():
    """Re-upserting the same path points the node at the new chunk set."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()

            n1, c1 = make_file("a.md", "v1", chunk_count=2)
            await store.upsert([(n1, c1)])

            # Different chunks for the same path
            n2 = FileNode(path="a.md", st_mtime=2.0)
            c2 = [FileChunk(id="a.md::new", path="a.md", text="v2 only", start_line=0, end_line=1)]
            n2.chunk_ids = [c.id for c in c2]
            await store.upsert([(n2, c2)])

            # The node now references the new chunk set, not the old one.
            nodes = await store.get_nodes(["a.md"])
            assert nodes[0].chunk_ids == ["a.md::new"]
            assert "a.md::new" in store.file_chunks

            await store.close()
            print("✓ test_upsert_replaces_old_chunks passed")

    asyncio.run(run())


def test_delete_by_path_single():
    """delete_by_path drops chunks and the node entry."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()

            await store.upsert([make_file("a.md", "alpha"), make_file("b.md", "beta")])
            await store.delete("a.md")

            assert all(c.path != "a.md" for c in store.file_chunks.values())
            assert {n.path for n in await store.get_nodes()} == {"b.md"}

            await store.close()
            print("✓ test_delete_by_path_single passed")

    asyncio.run(run())


def test_delete_by_path_list():
    """delete_by_path accepts a list of paths."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()

            await store.upsert(
                [
                    make_file("a.md", "alpha"),
                    make_file("b.md", "beta"),
                    make_file("c.md", "gamma"),
                ],
            )
            await store.delete(["a.md", "b.md"])

            assert {n.path for n in await store.get_nodes()} == {"c.md"}
            assert all(c.path == "c.md" for c in store.file_chunks.values())

            await store.close()
            print("✓ test_delete_by_path_list passed")

    asyncio.run(run())


def test_delete_by_path_missing_is_noop():
    """Deleting a nonexistent path is a no-op."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()

            await store.upsert([make_file("a.md", "alpha")])
            before = len(store.file_chunks)
            await store.delete("ghost.md")
            assert len(store.file_chunks) == before

            await store.close()
            print("✓ test_delete_by_path_missing_is_noop passed")

    asyncio.run(run())


def test_clear():
    """clear() empties chunks and the file graph."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()

            await store.upsert([make_file("a.md", "alpha"), make_file("b.md", "beta")])
            await store.clear()

            assert store.file_chunks == {}
            assert await store.get_nodes() == []

            await store.close()
            print("✓ test_clear passed")

    asyncio.run(run())


def test_keyword_search():
    """keyword_search returns matching chunks ranked by BM25 score."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()

            await store.upsert(
                [
                    make_file("a.md", "python programming language"),
                    make_file("b.md", "java programming language"),
                    make_file("c.md", "python data analysis"),
                ],
            )

            results = await store.keyword_search("python", limit=5, search_filter={})
            paths = {r.path for r in results}
            assert "a.md" in paths or "c.md" in paths
            # Each result should carry a keyword score.
            for r in results:
                assert r.scores.get("keyword", 0) > 0

            await store.close()
            print("✓ test_keyword_search passed")

    asyncio.run(run())


def test_keyword_search_empty_query():
    """Empty/whitespace queries return no results."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()
            await store.upsert([make_file("a.md", "hello")])

            assert await store.keyword_search("", limit=5, search_filter={}) == []
            assert await store.keyword_search("   ", limit=5, search_filter={}) == []

            await store.close()
            print("✓ test_keyword_search_empty_query passed")

    asyncio.run(run())


def test_vector_search_disabled_returns_empty():
    """Without an embedding model, vector_search returns []."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()
            await store.upsert([make_file("a.md", "hello")])

            assert store.embedding_model is None
            assert await store.vector_search("hello", limit=5, search_filter={}) == []

            await store.close()
            print("✓ test_vector_search_disabled_returns_empty passed")

    asyncio.run(run())


def test_persistence_roundtrip():
    """close() dumps chunks; a fresh store loads them from disk."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            s1 = await make_store()
            await s1.upsert([make_file("a.md", "alpha"), make_file("b.md", "beta")])
            await s1.close()

            s2 = await make_store()
            assert {c.path for c in s2.file_chunks.values()} == {"a.md", "b.md"}
            # Graph should also be persisted independently via its own dump.
            assert {n.path for n in await s2.get_nodes()} == {"a.md", "b.md"}
            await s2.close()
            print("✓ test_persistence_roundtrip passed")

    asyncio.run(run())


def test_rebuild_links_delegates_to_graph():
    """rebuild_links() on the store delegates to the underlying file_graph."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_store()

            from reme4.schema import FileLink

            node = FileNode(
                path="a.md",
                st_mtime=1.0,
                links=[FileLink(source_path="a.md", target_path="b.md")],
            )
            chunks = [FileChunk(id="a::1", path="a.md", text="x", start_line=0, end_line=1)]
            node.chunk_ids = [c.id for c in chunks]
            await store.upsert([(node, chunks), make_file("b.md", "beta")])

            await store.rebuild_links()
            inlinks = await store.get_inlinks("b.md")
            assert {lnk.source_path for lnk in inlinks} == {"a.md"}

            await store.close()
            print("✓ test_rebuild_links_delegates_to_graph passed")

    asyncio.run(run())


# --- FaissLocalFileStore tests --------------------------------------------------


def _skip_if_no_faiss(name: str) -> bool:
    if not _FAISS_AVAILABLE:
        print(f"⊘ {name} skipped (faiss not installed)")
        return True
    return False


async def make_faiss_store(store_name: str = "test_faiss", **kwargs) -> FaissLocalFileStore:
    """Build a started FaissLocalFileStore wired to FakeEmbeddingModel (no API calls)."""
    store = FaissLocalFileStore(store_name=store_name, embedding_model="fake", **kwargs)
    fake = FakeEmbeddingModel()
    # Replace the unresolved Dependency placeholder with a concrete instance and
    # let start() cascade lifecycle to it via _owned.
    store.embedding_model = fake
    store._owned.append(fake)
    await store.start()
    return store


def test_faiss_vector_search_basic():
    """vector_search returns chunks ranked by cosine similarity to the query."""
    if _skip_if_no_faiss("test_faiss_vector_search_basic"):
        return

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_faiss_store()
            await store.upsert(
                [
                    make_file("a.md", "alpha"),
                    make_file("b.md", "beta"),
                    make_file("c.md", "alpha gamma"),
                ],
            )

            results = await store.vector_search("alpha", limit=3, search_filter={})
            assert results, "vector_search returned no results"
            for r in results:
                assert "vector" in r.scores
                assert r.scores["score"] == r.scores["vector"]
            # Top hit should match an "alpha"-bearing doc.
            assert results[0].path in {"a.md", "c.md"}
            # All distinct chunks (no duplicates from tombstones).
            assert len({r.id for r in results}) == len(results)

            await store.close()
            print("✓ test_faiss_vector_search_basic passed")

    asyncio.run(run())


def test_faiss_persistence_roundtrip():
    """close() writes FAISS sidecar; a fresh store loads it without rebuilding."""
    if _skip_if_no_faiss("test_faiss_persistence_roundtrip"):
        return

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            s1 = await make_faiss_store()
            await s1.upsert([make_file("a.md", "alpha"), make_file("b.md", "beta")])
            r1 = await s1.vector_search("alpha", limit=2, search_filter={})
            assert s1.faiss_path.exists() is False  # not yet dumped
            await s1.close()
            assert s1.faiss_path.exists() and s1.faiss_idmap_path.exists()

            s2 = await make_faiss_store()
            assert s2._faiss_index is not None
            assert s2._faiss_index.ntotal == 2
            r2 = await s2.vector_search("alpha", limit=2, search_filter={})
            assert [r.path for r in r2] == [r.path for r in r1]
            await s2.close()
            print("✓ test_faiss_persistence_roundtrip passed")

    asyncio.run(run())


def test_faiss_delete_removes_from_search():
    """Deleting a file tombstones its chunks; subsequent search excludes them."""
    if _skip_if_no_faiss("test_faiss_delete_removes_from_search"):
        return

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_faiss_store()
            await store.upsert([make_file("a.md", "alpha"), make_file("b.md", "alpha beta")])

            await store.delete("a.md")
            results = await store.vector_search("alpha", limit=5, search_filter={})
            assert all(r.path != "a.md" for r in results)
            # All tombstoned rows still present in id_map; live mapping shrank.
            assert len(store._id_to_row) == 1
            assert len(store._tombstones) == 1

            await store.close()
            print("✓ test_faiss_delete_removes_from_search passed")

    asyncio.run(run())


def test_faiss_upsert_replaces_vectors():
    """Re-upserting a path with new chunk ids tombstones the old vectors."""
    if _skip_if_no_faiss("test_faiss_upsert_replaces_vectors"):
        return

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_faiss_store()
            n1, c1 = make_file("a.md", "alpha", chunk_count=2)
            await store.upsert([(n1, c1)])
            assert store._faiss_index.ntotal == 2
            assert len(store._tombstones) == 0

            # New chunk ids for the same path → old ones become tombstones.
            n2 = FileNode(path="a.md", st_mtime=2.0)
            c2 = [FileChunk(id="a.md::new", path="a.md", text="gamma", start_line=0, end_line=1)]
            n2.chunk_ids = [c.id for c in c2]
            await store.upsert([(n2, c2)])

            assert store._faiss_index.ntotal == 3  # 2 old + 1 new appended
            assert len(store._tombstones) == 2  # both old rows tombstoned
            assert "a.md::new" in store._id_to_row

            results = await store.vector_search("gamma", limit=5, search_filter={})
            assert results and results[0].id == "a.md::new"

            await store.close()
            print("✓ test_faiss_upsert_replaces_vectors passed")

    asyncio.run(run())


def test_faiss_clear_empties_index():
    """clear() resets FAISS state and removes sidecar files."""
    if _skip_if_no_faiss("test_faiss_clear_empties_index"):
        return

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = await make_faiss_store()
            await store.upsert([make_file("a.md", "alpha")])
            await store.dump()
            assert store.faiss_path.exists()

            await store.clear()
            assert store._faiss_index.ntotal == 0
            assert store._id_map == [] and store._id_to_row == {}
            assert not store.faiss_path.exists()
            assert not store.faiss_idmap_path.exists()
            assert await store.vector_search("alpha", limit=5, search_filter={}) == []

            await store.close()
            print("✓ test_faiss_clear_empties_index passed")

    asyncio.run(run())


def test_faiss_rebuild_when_sidecar_missing():
    """If the FAISS sidecar is missing on load, the index rebuilds from chunks JSONL."""
    if _skip_if_no_faiss("test_faiss_rebuild_when_sidecar_missing"):
        return

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            s1 = await make_faiss_store()
            await s1.upsert([make_file("a.md", "alpha"), make_file("b.md", "beta")])
            await s1.close()

            # Drop the FAISS sidecar but keep chunks JSONL — load() must rebuild.
            s1.faiss_path.unlink()
            s1.faiss_idmap_path.unlink()

            s2 = await make_faiss_store()
            assert s2._faiss_index is not None and s2._faiss_index.ntotal == 2
            results = await s2.vector_search("alpha", limit=2, search_filter={})
            assert any(r.path == "a.md" for r in results)
            await s2.close()
            print("✓ test_faiss_rebuild_when_sidecar_missing passed")

    asyncio.run(run())


def test_faiss_disabled_without_embedding():
    """embedding_model="" → FAISS path stays dormant; vector_search returns []."""
    if _skip_if_no_faiss("test_faiss_disabled_without_embedding"):
        return

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            store = FaissLocalFileStore(store_name="disabled", embedding_model="")
            await store.start()
            await store.upsert([make_file("a.md", "alpha")])

            assert store.embedding_model is None
            assert store._faiss_index is None
            assert await store.vector_search("alpha", limit=5, search_filter={}) == []

            await store.close()
            print("✓ test_faiss_disabled_without_embedding passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== LocalFileStore Tests ===")
    test_upsert_single_file()
    test_upsert_multiple_files()
    test_upsert_replaces_old_chunks()
    test_delete_by_path_single()
    test_delete_by_path_list()
    test_delete_by_path_missing_is_noop()
    test_clear()
    test_keyword_search()
    test_keyword_search_empty_query()
    test_vector_search_disabled_returns_empty()
    test_persistence_roundtrip()
    test_rebuild_links_delegates_to_graph()

    print("\n=== FaissLocalFileStore Tests ===")
    test_faiss_vector_search_basic()
    test_faiss_persistence_roundtrip()
    test_faiss_delete_removes_from_search()
    test_faiss_upsert_replaces_vectors()
    test_faiss_clear_empties_index()
    test_faiss_rebuild_when_sidecar_missing()
    test_faiss_disabled_without_embedding()

    print("\n所有测试通过!")
