"""Tests for reme4 common steps.

Two surfaces share this file:

* **In-process job tests** (top half) build an ``Application`` from the
  default config and call ``run_job`` directly — no subprocess, no HTTP.
  Each test uses an isolated cwd so the vault (``.reme`` by default)
  does not collide.
* **Direct unit tests** (bottom half) exercise ``TraverseStep``
  (registered as ``traverse_step``) — BFS over wikilink edges from a
  seed file, forward / backward / both — against a freshly built
  ``LocalFileStore`` (embedding disabled).
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings

from reme4 import Application, __version__ as REME_VERSION
from reme4.components.file_store import LocalFileStore
from reme4.config import resolve_app_config
from reme4.schema import FileLink, FileNode
from reme4.steps.index import traverse as traverse_mod
from reme4.utils import load_env

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


async def _make_app() -> Application:
    """Build and start an Application with the default config, logging silenced."""
    load_env()
    cfg = resolve_app_config(log_to_console=False, log_to_file=False, enable_logo=False)
    app = Application(**cfg)
    await app.start()
    return app


def _node(path: str, links: list[tuple[str, str | None, str | None]] | None = None) -> FileNode:
    """Build a FileNode with (target_path, target_anchor, predicate) outgoing edges."""
    return FileNode(
        path=path,
        st_mtime=1.0,
        links=[FileLink(source_path=path, target_path=t, target_anchor=a, predicate=p) for t, a, p in (links or [])],
    )


async def _make_store(nodes: list[FileNode]) -> LocalFileStore:
    """LocalFileStore seeded with the given graph nodes (no files on disk)."""
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    if nodes:
        await store.file_graph.upsert_nodes(nodes)
    return store


def _edges(step) -> list[dict]:
    return step.context.response.metadata.get("edges", [])


# ===========================================================================
# In-process job tests: version / help / health_check / search / reindex
# ===========================================================================


def test_version_job():
    """version job should return the package version string."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                resp = await app.run_job("version")
                assert resp.success is True
                assert resp.answer == REME_VERSION
                assert resp.metadata.get("version") == REME_VERSION
            finally:
                await app.close()
        print("✓ test_version_job passed")

    _run(run())


def test_help_job():
    """help job should list jobs except itself."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                resp = await app.run_job("help")
                assert resp.success is True
                assert isinstance(resp.answer, str)
                assert resp.metadata.get("job_count", 0) > 0
                assert "`help`" not in resp.answer
                for expected_job in ("version", "health_check", "search"):
                    assert expected_job in resp.answer, f"help missing {expected_job!r}: {resp.answer!r}"
            finally:
                await app.close()
        print("✓ test_help_job passed")

    _run(run())


def test_search_job_empty_store():
    """search on an empty store should return successfully with zero results."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                resp = await app.run_job("search", query="hello world", limit=5)
                assert resp.success is True
                counts = resp.metadata.get("counts", {})
                assert isinstance(counts, dict)
                assert counts.get("returned", -1) == 0
            finally:
                await app.close()
        print("✓ test_search_job_empty_store passed")

    _run(run())


def test_search_job_missing_query():
    """search with empty query returns success=False and a query-related error in answer."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                resp = await app.run_job("search", query="")
                assert resp.success is False
                assert "query" in str(resp.answer).lower()
            finally:
                await app.close()
        print("✓ test_search_job_missing_query passed")

    _run(run())


def test_all_jobs_single_app():
    """Run every common job against one shared in-process Application for efficiency."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                resp = await app.run_job("version")
                assert resp.answer == REME_VERSION

                resp = await app.run_job("help")
                assert resp.metadata.get("job_count", 0) > 0

                resp = await app.run_job("health_check")
                assert isinstance(resp.metadata.get("health"), dict)

                resp = await app.run_job("search", query="anything")
                assert resp.success is True

                resp = await app.run_job("reindex")
                assert isinstance(resp.metadata.get("counts"), dict)
            finally:
                await app.close()
        print("✓ test_all_jobs_single_app passed")

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
            results = _edges(step)
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
            results = _edges(step)
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
            results = _edges(step)
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
            payload = _edges(step)
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
            payload = _edges(step)
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
            results = _edges(step)
            assert {r["path"] for r in results} == {"upstream.md", "downstream.md"}
            await store.close()
        print("✓ test_traverse_both_directions passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== reme4 common steps in-process tests ===")
    test_version_job()
    test_help_job()
    test_search_job_empty_store()
    test_search_job_missing_query()
    test_all_jobs_single_app()
    print("\n=== traverse step tests ===")
    test_traverse_forward_depth_1()
    test_traverse_backward_returns_inlinks()
    test_traverse_depth_2_expands()
    test_traverse_short_seed_yields_empty()
    test_traverse_not_found_seed()
    test_traverse_both_directions()
    print("\n所有测试通过!")
