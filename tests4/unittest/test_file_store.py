"""Tests for LocalFileStore."""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings

from reme4.components.file_store import LocalFileStore
from reme4.schema import FileChunk, FileNode

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


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
    print("\n所有测试通过!")
