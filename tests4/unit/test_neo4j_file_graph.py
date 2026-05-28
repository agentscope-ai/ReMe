"""Tests for Neo4jFileGraph.

Skipped automatically if (a) the ``neo4j`` driver isn't installed,
or (b) a Neo4j instance isn't reachable at the configured URI.

Override the URI / auth via env vars:
    NEO4J_URI       (default ``bolt://localhost:7687``)
    NEO4J_USER      (default ``neo4j``)
    NEO4J_PASSWORD  (default ``neo4j``)
    NEO4J_DATABASE  (default ``neo4j``)
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile

import pytest

from reme4.schema import FileLink, FileNode


URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j")
DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


try:
    from reme4.components.file_graph import Neo4jFileGraph

    _NEO4J_IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover
    _NEO4J_IMPORT_ERROR = e


async def _probe_neo4j() -> str | None:
    """Try to connect; return None if OK, error string if unreachable."""
    if _NEO4J_IMPORT_ERROR is not None:
        return f"import failed: {_NEO4J_IMPORT_ERROR}"
    try:
        from neo4j import AsyncGraphDatabase
    except ImportError as e:
        return f"neo4j driver not installed: {e}"
    try:
        driver = AsyncGraphDatabase.driver(URI, auth=(USER, PASSWORD))
        async with driver.session(database=DATABASE) as session:
            await session.run("RETURN 1")
        await driver.close()
        return None
    except Exception as e:  # pragma: no cover
        return f"connect failed: {e}"


_PROBE_NOT_RUN = object()
_PROBE_REASON: str | None | object = _PROBE_NOT_RUN  # sentinel: "not probed"


def _probe_reason() -> str | None:
    """Probe Neo4j once per process; cache the outcome."""
    global _PROBE_REASON
    if _PROBE_REASON is _PROBE_NOT_RUN:
        _PROBE_REASON = asyncio.run(_probe_neo4j())
    return _PROBE_REASON  # type: ignore[return-value]


pytestmark = pytest.mark.skipif(
    _probe_reason() is not None,
    reason=f"Neo4j unavailable: {_probe_reason()}",
)


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
    """Build a FileNode with outgoing (target_path, target_anchor) pairs."""
    return FileNode(
        path=path,
        st_mtime=1.0,
        links=[FileLink(source_path=path, target_path=t, target_anchor=a) for t, a in (links or [])],
    )


async def _fresh_graph() -> "Neo4jFileGraph":  # type: ignore[name-defined]
    """Build a started Neo4jFileGraph wiped clean."""
    graph = Neo4jFileGraph(uri=URI, user=USER, password=PASSWORD, database=DATABASE)
    await graph.start()
    await graph.clear()
    return graph


def test_upsert_and_get_nodes():
    """upsert_nodes stores; get_nodes returns by paths or all."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _fresh_graph()
            try:
                await graph.upsert_nodes(
                    [make_node("a.md", [("b.md", None)]), make_node("b.md")],
                )
                got_all = await graph.get_nodes()
                assert {n.path for n in got_all} == {"a.md", "b.md"}
                got_one = await graph.get_nodes(["a.md"])
                assert len(got_one) == 1 and got_one[0].path == "a.md"
                assert await graph.get_nodes(["nope.md"]) == []
                assert await graph.get_nodes([]) == []
            finally:
                await graph.clear()
                await graph.close()
        print("✓ test_upsert_and_get_nodes passed")

    asyncio.run(run())


def test_outlinks_skip_virtual_targets():
    """get_outlinks excludes edges into virtual placeholder nodes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _fresh_graph()
            try:
                await graph.upsert_nodes(
                    [
                        make_node("a.md", [("b.md", None), ("ghost.md", None)]),
                        make_node("b.md"),
                    ],
                )
                outs = await graph.get_outlinks("a.md")
                assert {link.target_path for link in outs} == {"b.md"}
                for link in outs:
                    assert link.source_path == "a.md"
            finally:
                await graph.clear()
                await graph.close()
        print("✓ test_outlinks_skip_virtual_targets passed")

    asyncio.run(run())


def test_inlinks_carry_source_path():
    """get_inlinks returns FileLinks whose source_path is the linking node."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _fresh_graph()
            try:
                await graph.upsert_nodes(
                    [
                        make_node("a.md", [("b.md", "anchor1")]),
                        make_node("c.md", [("b.md", None)]),
                        make_node("b.md"),
                    ],
                )
                ins = await graph.get_inlinks("b.md")
                sources = {link.source_path for link in ins}
                assert sources == {"a.md", "c.md"}
                # Each link's target should be the queried path.
                for link in ins:
                    assert link.target_path == "b.md"
            finally:
                await graph.clear()
                await graph.close()
        print("✓ test_inlinks_carry_source_path passed")

    asyncio.run(run())


def test_delete_demotes_then_repromotes():
    """delete_nodes makes a node virtual; re-upsert promotes pending edges back."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _fresh_graph()
            try:
                await graph.upsert_nodes(
                    [make_node("a.md", [("b.md", None)]), make_node("b.md")],
                )
                assert {link.source_path for link in await graph.get_inlinks("b.md")} == {"a.md"}

                await graph.delete_nodes(["b.md"])
                assert await graph.get_nodes(["b.md"]) == []
                # a's outlink is hidden because b is now virtual.
                assert await graph.get_outlinks("a.md") == []

                await graph.upsert_nodes([make_node("b.md")])
                assert {link.source_path for link in await graph.get_inlinks("b.md")} == {"a.md"}
            finally:
                await graph.clear()
                await graph.close()
        print("✓ test_delete_demotes_then_repromotes passed")

    asyncio.run(run())


def test_rebuild_links_idempotent():
    """rebuild_links reconstructs identical out/in views from per-node payloads."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _fresh_graph()
            try:
                await graph.upsert_nodes(
                    [
                        make_node("a.md", [("b.md", None), ("c.md", "h")]),
                        make_node("b.md"),
                        make_node("c.md"),
                    ],
                )
                before_out = sorted((link.target_path, link.target_anchor) for link in await graph.get_outlinks("a.md"))
                before_in_b = sorted(link.source_path for link in await graph.get_inlinks("b.md"))

                await graph.rebuild_links()

                after_out = sorted((link.target_path, link.target_anchor) for link in await graph.get_outlinks("a.md"))
                after_in_b = sorted(link.source_path for link in await graph.get_inlinks("b.md"))
                assert before_out == after_out
                assert before_in_b == after_in_b
            finally:
                await graph.clear()
                await graph.close()
        print("✓ test_rebuild_links_idempotent passed")

    asyncio.run(run())


def test_clear_wipes_everything():
    """clear() drops every node and edge in the database."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _fresh_graph()
            try:
                await graph.upsert_nodes(
                    [make_node("a.md", [("b.md", None)]), make_node("b.md")],
                )
                await graph.clear()
                assert await graph.get_nodes() == []
            finally:
                await graph.close()
        print("✓ test_clear_wipes_everything passed")

    asyncio.run(run())


def test_node_roundtrip_preserves_frontmatter_and_links():
    """Upsert → get_nodes round-trip preserves frontmatter + links + chunk_ids."""

    async def run():
        from reme4.schema.file_node import FileFrontMatter

        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            graph = await _fresh_graph()
            try:
                node = FileNode(
                    path="topics/Alice.md",
                    st_mtime=1234.5,
                    links=[
                        FileLink(
                            source_path="topics/Alice.md",
                            target_path="topics/Bob.md",
                            target_anchor="intro",
                            predicate="knows",
                        ),
                    ],
                    chunk_ids=["chunk-a1", "chunk-a2", "chunk-a3"],
                    front_matter=FileFrontMatter(
                        name="Alice",
                        description="a person",
                    ),
                )
                await graph.upsert_nodes([node, make_node("topics/Bob.md")])
                got = await graph.get_nodes(["topics/Alice.md"])
                assert len(got) == 1
                back = got[0]
                assert back.path == "topics/Alice.md"
                assert back.st_mtime == 1234.5
                assert back.front_matter.name == "Alice"
                assert back.front_matter.description == "a person"
                assert back.chunk_ids == ["chunk-a1", "chunk-a2", "chunk-a3"]
                assert len(back.links) == 1
                link = back.links[0]
                assert link.source_path == "topics/Alice.md"
                assert link.target_path == "topics/Bob.md"
                assert link.target_anchor == "intro"
                assert link.predicate == "knows"

                # An upsert with empty chunk_ids should also round-trip cleanly
                # (and overwrite the previous list).
                await graph.upsert_nodes(
                    [
                        FileNode(
                            path="topics/Alice.md",
                            st_mtime=1234.5,
                            chunk_ids=[],
                        ),
                    ],
                )
                got2 = await graph.get_nodes(["topics/Alice.md"])
                assert got2 and got2[0].chunk_ids == []
            finally:
                await graph.clear()
                await graph.close()
        print("✓ test_node_roundtrip_preserves_frontmatter_and_links passed")

    asyncio.run(run())


if __name__ == "__main__":
    if _probe_reason() is not None:
        print(f"Skipping Neo4j tests: {_probe_reason()}")
    else:
        print("\n=== Neo4jFileGraph tests ===")
        test_upsert_and_get_nodes()
        test_outlinks_skip_virtual_targets()
        test_inlinks_carry_source_path()
        test_delete_demotes_then_repromotes()
        test_rebuild_links_idempotent()
        test_clear_wipes_everything()
        test_node_roundtrip_preserves_frontmatter_and_links()
        print("\n所有测试通过!")
