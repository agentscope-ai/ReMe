"""Tests for FileCatalog backends."""

# pylint: disable=protected-access

import asyncio
import os
import tempfile

import pytest

from reme4.components.file_catalog import LocalFileCatalog
from reme4.schema import FileNode


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


def make_node(path: str, mtime: float = 1.0) -> FileNode:
    """Build a FileNode with the given path and mtime for fixture use."""
    return FileNode(path=path, st_mtime=mtime)


# All backends should satisfy the same BaseFileCatalog contract.
BACKENDS = [LocalFileCatalog]


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_upsert_and_get_nodes(backend_cls):
    """upsert stores nodes; get_nodes returns by path or all."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            catalog = backend_cls()
            await catalog.start()

            await catalog.upsert([make_node("a.md"), make_node("b.md")])

            got_all = await catalog.get_nodes()
            assert {n.path for n in got_all} == {"a.md", "b.md"}

            got_one = await catalog.get_nodes(["a.md"])
            assert len(got_one) == 1
            assert got_one[0].path == "a.md"

            got_missing = await catalog.get_nodes(["nope.md"])
            assert got_missing == []

            await catalog.close()
            print(f"✓ test_upsert_and_get_nodes[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_upsert_replaces_existing(backend_cls):
    """Re-upserting a node with the same path overwrites the prior entry."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            catalog = backend_cls()
            await catalog.start()

            await catalog.upsert([make_node("a.md", mtime=1.0)])
            await catalog.upsert([make_node("a.md", mtime=2.0)])

            nodes = await catalog.get_nodes(["a.md"])
            assert len(nodes) == 1
            assert nodes[0].st_mtime == 2.0

            await catalog.close()
            print(f"✓ test_upsert_replaces_existing[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_delete_single_and_list(backend_cls):
    """delete accepts both a single path and a list; missing paths are no-ops."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            catalog = backend_cls()
            await catalog.start()

            await catalog.upsert([make_node("a.md"), make_node("b.md"), make_node("c.md")])

            await catalog.delete("a.md")
            assert {n.path for n in await catalog.get_nodes()} == {"b.md", "c.md"}

            await catalog.delete(["b.md", "ghost.md"])
            assert {n.path for n in await catalog.get_nodes()} == {"c.md"}

            await catalog.close()
            print(f"✓ test_delete_single_and_list[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_get_nodes_empty_inputs(backend_cls):
    """get_nodes([]) returns []; get_nodes(None) returns all."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            catalog = backend_cls()
            await catalog.start()

            await catalog.upsert([make_node("a.md")])
            assert await catalog.get_nodes([]) == []
            assert len(await catalog.get_nodes(None)) == 1

            await catalog.close()
            print(f"✓ test_get_nodes_empty_inputs[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_persistence_roundtrip(backend_cls):
    """close() dumps; a fresh instance loads the same nodes from disk."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            c1 = backend_cls()
            await c1.start()
            await c1.upsert([make_node("a.md", mtime=10.0), make_node("b.md", mtime=20.0)])
            await c1.close()

            c2 = backend_cls()
            await c2.start()
            nodes = sorted(await c2.get_nodes(), key=lambda n: n.path)
            assert [n.path for n in nodes] == ["a.md", "b.md"]
            assert [n.st_mtime for n in nodes] == [10.0, 20.0]
            await c2.close()
            print(f"✓ test_persistence_roundtrip[{backend_cls.__name__}] passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== FileCatalog Tests ===")
    for backend in BACKENDS:
        test_upsert_and_get_nodes(backend)
        test_upsert_replaces_existing(backend)
        test_delete_single_and_list(backend)
        test_get_nodes_empty_inputs(backend)
        test_persistence_roundtrip(backend)
    print("\n所有测试通过!")
