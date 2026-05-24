"""Tests for reme4 common steps.

Two surfaces share this file:

* **HTTP / MCP E2E tests** (top half) spawn ``reme4 start`` via
  ``mock_reme_server`` and drive ``version`` / ``help`` / ``search`` /
  ``init`` / ``demo`` over the wire. Each test uses an isolated cwd so
  the vault (``.reme`` by default) does not collide.
* **Direct unit tests** (bottom half) exercise ``TraverseStep``
  (registered as ``traverse_step``) — BFS over wikilink edges from a
  seed file, forward / backward / both — against a freshly built
  ``LocalFileStore`` (embedding disabled).
"""

# pylint: disable=protected-access

import asyncio
import json
import os
import tempfile
import warnings

from reme4 import __version__ as REME_VERSION
from reme4.components.file_store import LocalFileStore
from reme4.schema import FileLink, FileNode
from reme4.steps.common import traverse as traverse_mod
from reme4.utils import call_action, call_and_check, mock_reme_server

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class _temp_chdir:
    """chdir to path for the duration of the block; restore on exit."""

    def __init__(self, path):
        self.path = path
        self._old = None

    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self._old)


def _run(coro):
    """Run an async coroutine on a fresh isolated event loop."""
    asyncio.run(coro)


def _node(path: str, links: list[tuple[str, str | None, str | None]] | None = None) -> FileNode:
    """Build a FileNode with (target_path, target_anchor, predicate) outgoing edges."""
    return FileNode(
        path=path,
        st_mtime=1.0,
        links=[FileLink(source_path=path, target_path=t, target_anchor=a, predicate=p) for t, a, p in (links or [])],
    )


async def _make_store(nodes: list[FileNode]) -> LocalFileStore:
    """LocalFileStore seeded with the given graph nodes (no files on disk)."""
    store = LocalFileStore(store_name="t", embedding_model="")
    await store.start()
    if nodes:
        await store.file_graph.upsert_nodes(nodes)
    return store


def _answer(step) -> object:
    return json.loads(step.context.response.answer)


# ===========================================================================
# HTTP / MCP E2E tests: version / help / search / init / demo
# ===========================================================================


def test_version_job():
    """version job should return the package version string."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "version",
                    host=host,
                    port=port,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and r.get("answer") == REME_VERSION
                        and r.get("metadata", {}).get("version") == REME_VERSION
                    ),
                )
        print("✓ test_version_job passed")

    _run(run())


def test_help_job():
    """help job should list jobs except itself."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                result = await call_and_check(
                    "help",
                    host=host,
                    port=port,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and isinstance(r.get("answer"), str)
                        and r.get("metadata", {}).get("job_count", 0) > 0
                        and "`help`" not in r["answer"]
                    ),
                )
                # Spot-check that a couple of known jobs appear in the listing.
                answer = result["answer"]
                for expected_job in ("version", "health_check", "search"):
                    if expected_job not in answer:
                        raise AssertionError(f"help output missing job {expected_job!r}: {answer!r}")
        print("✓ test_help_job passed")

    _run(run())


def test_search_job_empty_store():
    """search on an empty store should return successfully with zero results."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                await call_and_check(
                    "search",
                    host=host,
                    port=port,
                    query="hello world",
                    limit=5,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and isinstance(r.get("metadata"), dict)
                        and isinstance(r["metadata"].get("counts"), dict)
                        and r["metadata"]["counts"].get("returned", -1) == 0
                    ),
                )
        print("✓ test_search_job_empty_store passed")

    _run(run())


def test_search_job_missing_query():
    """search without a query should surface the assertion error in `answer`."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                result = await call_action("search", host=host, port=port, query="")
                if not isinstance(result, dict):
                    raise AssertionError(f"expected dict response, got {result!r}")
                if "query" not in str(result.get("answer", "")).lower():
                    raise AssertionError(f"expected query-related error in answer, got {result!r}")
        print("✓ test_search_job_missing_query passed")

    _run(run())


# -- aggregate: reuse one server instance for all jobs -------------------


def test_all_jobs_one_server():
    """Run every common job against a single shared server for efficiency."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            async with mock_reme_server() as (host, port):
                # version
                await call_and_check(
                    "version",
                    host=host,
                    port=port,
                    validator=lambda r: isinstance(r, dict) and r.get("answer") == REME_VERSION,
                )
                # help
                await call_and_check(
                    "help",
                    host=host,
                    port=port,
                    validator=lambda r: isinstance(r, dict) and r.get("metadata", {}).get("job_count", 0) > 0,
                )
                # health_check
                await call_and_check(
                    "health_check",
                    host=host,
                    port=port,
                    validator=lambda r: isinstance(r, dict)
                    and isinstance(
                        r.get("metadata", {}).get("health"),
                        dict,
                    ),
                )
                # search (empty store)
                await call_and_check(
                    "search",
                    host=host,
                    port=port,
                    query="anything",
                    validator=lambda r: isinstance(r, dict) and r.get("success") is True,
                )
                # reindex
                await call_and_check(
                    "reindex",
                    host=host,
                    port=port,
                    validator=lambda r: isinstance(r, dict) and isinstance(r.get("metadata", {}).get("counts"), dict),
                )
        print("✓ test_all_jobs_one_server passed")

    _run(run())


# ===========================================================================
# Direct unit tests: TraverseStep
# (LocalFileStore, no HTTP server — BFS over wikilink edges)
# ===========================================================================


def test_traverse_forward_depth_1():
    """depth=1 forward returns direct outbound neighbors."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("a.md", [("b.md", None, None), ("c.md", "intro", "ref")]),
                    _node("b.md"),
                    _node("c.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="a.md", direction="forward", depth=1)
            results = _answer(step)
            paths = {r["path"] for r in results}
            assert paths == {"b.md", "c.md"}
            # The 'ref' edge should report its predicate/anchor.
            c_edge = next(r for r in results if r["path"] == "c.md")
            assert c_edge["predicate"] == "ref"
            assert c_edge["anchor"] == "intro"
            assert c_edge["via"] == "a.md"
            assert c_edge["depth"] == 1
            await store.close()
        print("✓ test_traverse_forward_depth_1 passed")

    asyncio.run(run())


def test_traverse_backward_returns_inlinks():
    """direction=backward walks inbound edges."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("a.md", [("b.md", None, None)]),
                    _node("c.md", [("b.md", None, None)]),
                    _node("b.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="b.md", direction="backward", depth=1)
            results = _answer(step)
            assert {r["path"] for r in results} == {"a.md", "c.md"}
            await store.close()
        print("✓ test_traverse_backward_returns_inlinks passed")

    asyncio.run(run())


def test_traverse_depth_2_expands():
    """depth=2 traverses one hop beyond direct neighbors."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("a.md", [("b.md", None, None)]),
                    _node("b.md", [("c.md", None, None)]),
                    _node("c.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="a.md", direction="forward", depth=2)
            results = _answer(step)
            depth_map = {r["path"]: r["depth"] for r in results}
            assert depth_map.get("b.md") == 1
            assert depth_map.get("c.md") == 2
            await store.close()
        print("✓ test_traverse_depth_2_expands passed")

    asyncio.run(run())


def test_traverse_short_seed_yields_empty():
    """A short (not relative to the vault) seed isn't resolved anymore — BFS simply
    finds no edges from a path that doesn't match any graph node."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("topics/Bob.md"),
                    _node("people/Bob.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="Bob", direction="forward", depth=1)
            payload = _answer(step)
            # No error, just empty results because "Bob" isn't a graph key.
            assert payload == []
            await store.close()
        print("✓ test_traverse_short_seed_yields_empty passed")

    asyncio.run(run())


def test_traverse_not_found_seed():
    """A seed not in the graph returns an empty list (no error)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store([_node("a.md")])
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="topics/ghost.md", direction="forward", depth=1)
            payload = _answer(step)
            assert payload == []
            await store.close()
        print("✓ test_traverse_not_found_seed passed")

    asyncio.run(run())


def test_traverse_both_directions():
    """direction=both walks out- and in-bound; depth=1 returns one hop in each direction."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("upstream.md", [("center.md", None, None)]),
                    _node("center.md", [("downstream.md", None, None)]),
                    _node("downstream.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="center.md", direction="both", depth=1)
            results = _answer(step)
            assert {r["path"] for r in results} == {"upstream.md", "downstream.md"}
            await store.close()
        print("✓ test_traverse_both_directions passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== reme4 common steps E2E tests ===")
    test_version_job()
    test_help_job()
    test_search_job_empty_store()
    test_search_job_missing_query()
    test_all_jobs_one_server()
    print("\n=== traverse step tests ===")
    test_traverse_forward_depth_1()
    test_traverse_backward_returns_inlinks()
    test_traverse_depth_2_expands()
    test_traverse_short_seed_yields_empty()
    test_traverse_not_found_seed()
    test_traverse_both_directions()
    print("\n所有测试通过!")
