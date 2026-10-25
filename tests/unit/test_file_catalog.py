"""Tests for FileCatalog backends."""

# pylint: disable=protected-access

from contextlib import chdir as temp_chdir
import tempfile

import pytest

from reme.components.file_catalog import LocalFileCatalog
from reme.schema import FileNode
from reme.utils.jsonl_zst import write_jsonl_zst


def make_node(path: str, mtime: float = 1.0) -> FileNode:
    """Build a FileNode with the given path and mtime for fixture use."""
    return FileNode(path=path, st_mtime=mtime)


# All backends should satisfy the same BaseFileCatalog contract.
BACKENDS = [LocalFileCatalog]


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_upsert_and_get_nodes(backend_cls):
    """upsert stores nodes; get_nodes returns by path or all."""

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


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_upsert_replaces_existing(backend_cls):
    """Re-upserting a node with the same path overwrites the prior entry."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        catalog = backend_cls()
        await catalog.start()

        await catalog.upsert([make_node("a.md", mtime=1.0)])
        await catalog.upsert([make_node("a.md", mtime=2.0)])

        nodes = await catalog.get_nodes(["a.md"])
        assert len(nodes) == 1
        assert nodes[0].st_mtime == 2.0

        await catalog.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_delete_single_and_list(backend_cls):
    """delete accepts both a single path and a list; missing paths are no-ops."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        catalog = backend_cls()
        await catalog.start()

        await catalog.upsert([make_node("a.md"), make_node("b.md"), make_node("c.md")])

        await catalog.delete("a.md")
        assert {n.path for n in await catalog.get_nodes()} == {"b.md", "c.md"}

        await catalog.delete(["b.md", "ghost.md"])
        assert {n.path for n in await catalog.get_nodes()} == {"c.md"}

        await catalog.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_get_nodes_empty_inputs(backend_cls):
    """get_nodes([]) returns []; get_nodes(None) returns all."""

    with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
        catalog = backend_cls()
        await catalog.start()

        await catalog.upsert([make_node("a.md")])
        assert await catalog.get_nodes([]) == []
        assert len(await catalog.get_nodes(None)) == 1

        await catalog.close()


@pytest.mark.parametrize("backend_cls", BACKENDS)
@pytest.mark.asyncio
async def test_persistence_roundtrip(backend_cls):
    """close() dumps; a fresh instance loads the same nodes from disk."""

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


@pytest.mark.asyncio
async def test_start_failure_does_not_overwrite_persisted_catalog(tmp_path):
    """A partial load must not dump incomplete in-memory state during rollback."""

    with temp_chdir(tmp_path):
        catalog = LocalFileCatalog()
        original_lines = [
            make_node("a.md").model_dump_json(),
            "not valid json",
            make_node("b.md").model_dump_json(),
        ]
        write_jsonl_zst(catalog._catalog_file, original_lines)
        original_bytes = catalog._catalog_file.read_bytes()

        with pytest.raises(ValueError):
            await catalog.start()

        assert catalog.is_started is False
        assert catalog._catalog_file.read_bytes() == original_bytes
