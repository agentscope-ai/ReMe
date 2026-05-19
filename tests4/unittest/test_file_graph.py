"""Tests for FileGraph backends (LocalFileGraph + NxFileGraph)."""

# pylint: disable=protected-access

import asyncio
import os
import tempfile

import pytest

from reme4.components.file_graph import LocalFileGraph, NxFileGraph
from reme4.schema import FileLink, FileNode


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


def make_node(path: str, links: list[tuple[str, str | None]] | None = None) -> FileNode:
    """Build a FileNode with the given outgoing (target_path, target_anchor) pairs."""
    return FileNode(
        path=path,
        st_mtime=1.0,
        links=[FileLink(source_path=path, target_path=t, target_anchor=a) for t, a in (links or [])],
    )


# Both backends should satisfy the same BaseFileGraph contract.
BACKENDS = [LocalFileGraph, NxFileGraph]


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_upsert_and_get_nodes(backend_cls):
    """upsert_nodes stores nodes; get_nodes returns them by path or all."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            graph = backend_cls()
            await graph.start()

            n1 = make_node("a.md", [("b.md", None)])
            n2 = make_node("b.md")
            await graph.upsert_nodes([n1, n2])

            got_all = await graph.get_nodes()
            assert {n.path for n in got_all} == {"a.md", "b.md"}

            got_one = await graph.get_nodes(["a.md"])
            assert len(got_one) == 1
            assert got_one[0].path == "a.md"

            got_missing = await graph.get_nodes(["nope.md"])
            assert got_missing == []

            await graph.close()
            print(f"✓ test_upsert_and_get_nodes[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_upsert_replaces_old_links(backend_cls):
    """Re-upserting a node with new links replaces the old outgoing edges."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            graph = backend_cls()
            await graph.start()

            await graph.upsert_nodes(
                [
                    make_node("a.md", [("b.md", None)]),
                    make_node("b.md"),
                    make_node("c.md"),
                ],
            )
            assert {lnk.target_path for lnk in await graph.get_outlinks("a.md")} == {"b.md"}

            # Replace a's link target from b → c
            await graph.upsert_nodes([make_node("a.md", [("c.md", None)])])
            assert {lnk.target_path for lnk in await graph.get_outlinks("a.md")} == {"c.md"}
            # b should no longer have a as an inlink
            assert await graph.get_inlinks("b.md") == []
            assert {lnk.source_path for lnk in await graph.get_inlinks("c.md")} == {"a.md"}

            await graph.close()
            print(f"✓ test_upsert_replaces_old_links[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_outlinks_skip_virtual_targets(backend_cls):
    """get_outlinks only returns links pointing to real (existing) nodes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            graph = backend_cls()
            await graph.start()

            # a links to b (real) and ghost (virtual)
            await graph.upsert_nodes(
                [
                    make_node("a.md", [("b.md", None), ("ghost.md", None)]),
                    make_node("b.md"),
                ],
            )

            outs = await graph.get_outlinks("a.md")
            targets = {lnk.target_path for lnk in outs}
            assert targets == {"b.md"}

            await graph.close()
            print(f"✓ test_outlinks_skip_virtual_targets[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_inlinks_promotion_after_upsert(backend_cls):
    """Edges to virtual targets become real inlinks once the target is upserted."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            graph = backend_cls()
            await graph.start()

            # b doesn't exist yet — link is pending
            await graph.upsert_nodes([make_node("a.md", [("b.md", None)])])
            assert await graph.get_inlinks("b.md") == []  # b not real yet

            # Now create b — pending edge promotes
            await graph.upsert_nodes([make_node("b.md")])
            inlinks = await graph.get_inlinks("b.md")
            assert {lnk.source_path for lnk in inlinks} == {"a.md"}

            await graph.close()
            print(f"✓ test_inlinks_promotion_after_upsert[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_delete_node_demotes_inbound(backend_cls):
    """Deleting a node makes it virtual; sources still hold the link, but get_inlinks([deleted]) is []."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            graph = backend_cls()
            await graph.start()

            await graph.upsert_nodes(
                [
                    make_node("a.md", [("b.md", None)]),
                    make_node("b.md"),
                ],
            )
            assert {lnk.source_path for lnk in await graph.get_inlinks("b.md")} == {"a.md"}

            await graph.delete_nodes(["b.md"])
            # b is no longer a real node
            assert await graph.get_nodes(["b.md"]) == []
            # inlinks query for a non-real node returns []
            assert await graph.get_inlinks("b.md") == []
            # a's outlink to b is hidden because b is virtual
            assert await graph.get_outlinks("a.md") == []

            # Re-upsert b — pending should re-promote
            await graph.upsert_nodes([make_node("b.md")])
            assert {lnk.source_path for lnk in await graph.get_inlinks("b.md")} == {"a.md"}

            await graph.close()
            print(f"✓ test_delete_node_demotes_inbound[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_delete_outgoing_links_cleared(backend_cls):
    """Deleting a source node drops its outgoing edges (no inlink left on its targets)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            graph = backend_cls()
            await graph.start()

            await graph.upsert_nodes(
                [
                    make_node("a.md", [("b.md", None)]),
                    make_node("b.md"),
                ],
            )
            await graph.delete_nodes(["a.md"])

            assert await graph.get_inlinks("b.md") == []

            await graph.close()
            print(f"✓ test_delete_outgoing_links_cleared[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_clear(backend_cls):
    """clear() removes all nodes and edges."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            graph = backend_cls()
            await graph.start()

            await graph.upsert_nodes(
                [
                    make_node("a.md", [("b.md", None)]),
                    make_node("b.md"),
                ],
            )
            await graph.clear()
            assert await graph.get_nodes() == []

            await graph.close()
            print(f"✓ test_clear[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_rebuild_links_idempotent(backend_cls):
    """rebuild_links produces the same outlink/inlink view as the original upserts."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            graph = backend_cls()
            await graph.start()

            await graph.upsert_nodes(
                [
                    make_node("a.md", [("b.md", None), ("c.md", "h")]),
                    make_node("b.md"),
                    make_node("c.md"),
                ],
            )

            before_out = sorted((lnk.target_path, lnk.target_anchor) for lnk in await graph.get_outlinks("a.md"))
            before_in = sorted(lnk.source_path for lnk in await graph.get_inlinks("b.md"))

            await graph.rebuild_links()

            after_out = sorted((lnk.target_path, lnk.target_anchor) for lnk in await graph.get_outlinks("a.md"))
            after_in = sorted(lnk.source_path for lnk in await graph.get_inlinks("b.md"))

            assert before_out == after_out
            assert before_in == after_in

            await graph.close()
            print(f"✓ test_rebuild_links_idempotent[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_persistence_roundtrip(backend_cls):
    """close() dumps; a fresh instance loads the same nodes from disk."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            g1 = backend_cls()
            await g1.start()
            await g1.upsert_nodes(
                [
                    make_node("a.md", [("b.md", None)]),
                    make_node("b.md"),
                ],
            )
            await g1.close()  # triggers dump

            g2 = backend_cls()
            await g2.start()  # triggers load
            paths = sorted(n.path for n in await g2.get_nodes())
            assert paths == ["a.md", "b.md"]
            # Inlink relationship should also be reconstructable.
            assert {lnk.source_path for lnk in await g2.get_inlinks("b.md")} == {"a.md"}
            await g2.close()
            print(f"✓ test_persistence_roundtrip[{backend_cls.__name__}] passed")

    asyncio.run(run())


@pytest.mark.parametrize("backend_cls", BACKENDS)
def test_get_nodes_empty_inputs(backend_cls):
    """get_nodes([]) returns []; get_nodes(None) returns all."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            graph = backend_cls()
            await graph.start()

            await graph.upsert_nodes([make_node("a.md")])
            assert await graph.get_nodes([]) == []
            assert len(await graph.get_nodes(None)) == 1

            await graph.close()
            print(f"✓ test_get_nodes_empty_inputs[{backend_cls.__name__}] passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== FileGraph Tests ===")
    for backend in BACKENDS:
        test_upsert_and_get_nodes(backend)
        test_upsert_replaces_old_links(backend)
        test_outlinks_skip_virtual_targets(backend)
        test_inlinks_promotion_after_upsert(backend)
        test_delete_node_demotes_inbound(backend)
        test_delete_outgoing_links_cleared(backend)
        test_clear(backend)
        test_rebuild_links_idempotent(backend)
        test_persistence_roundtrip(backend)
        test_get_nodes_empty_inputs(backend)
    print("\n所有测试通过!")
