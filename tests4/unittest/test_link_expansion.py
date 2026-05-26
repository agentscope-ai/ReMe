"""Tests for first-order neighbor link expansion.

Covers:
- ``link_expansion.get_first_order_neighbors_batch`` (shared module used by both
  search_step and read_step)
- ``link_expansion.get_first_order_neighbors`` (single-path wrapper)
- ``read_step`` rendering & metadata serialization helpers
- ``read_step`` with_neighbors end-to-end (full HTTP stack)
- ``search_step`` expand_links still functions after the refactor (regression)

Strategy: seed a ``LocalFileStore`` directly with synthetic ``FileNode`` /
``FileLink`` objects so we don't depend on the markdown parser or the
file_graph wiring. This isolates the neighbor-expansion logic from upstream
indexing details.
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import time
import warnings
from pathlib import Path

from reme4.components.file_store import LocalFileStore
from reme4.schema import FileChunk, FileFrontMatter, FileLink, FileNode
from reme4.steps.common.link_expansion import (
    get_first_order_neighbors,
    get_first_order_neighbors_batch,
)
from reme4.steps.crud.read import _neighbors_to_serializable, _render_neighbor_block
from reme4.utils import call_action, call_and_check, mock_reme_server

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _temp_chdir:
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
    asyncio.run(coro)


def _node(path: str, links: list[FileLink], **front_matter_fields) -> FileNode:
    """Build a FileNode with stable mtime and zero chunks."""
    fm = FileFrontMatter(**front_matter_fields) if front_matter_fields else FileFrontMatter()
    return FileNode(path=path, st_mtime=time.time(), links=links, chunk_ids=[], front_matter=fm)


def _link(src: str, tgt: str, *, anchor: str | None = None, predicate: str | None = None) -> FileLink:
    return FileLink(source_path=src, target_path=tgt, target_anchor=anchor, predicate=predicate)


async def _make_store_with_graph(items: list[tuple[FileNode, list[FileChunk]]]) -> LocalFileStore:
    """Create + start a LocalFileStore, wipe persisted state, upsert ``items``.

    The LocalFileGraph persists to a process-wide ``metadata/file_graph/default_v1.jsonl``
    that other tests share — we call ``clear()`` first to avoid contamination
    from prior runs.
    """
    fs = LocalFileStore(store_name="test_link_expansion_store", embedding_model="")
    await fs.start()
    await fs.clear()
    if items:
        await fs.upsert(items)
        await fs.rebuild_links()
    return fs


# ---------------------------------------------------------------------------
# Direct link_expansion tests
# ---------------------------------------------------------------------------


def test_get_neighbors_batch_basic_chain():
    """A → B → C: query for B yields outlinks=[C], inlinks=[A]."""

    async def run():
        items = [
            (_node("A.md", [_link("A.md", "B.md")]), []),
            (_node("B.md", [_link("B.md", "C.md")]), []),
            (_node("C.md", []), []),
        ]
        fs = await _make_store_with_graph(items)
        try:
            result = await get_first_order_neighbors_batch(fs, ["B.md"])
            assert "B.md" in result
            entry = result["B.md"]
            out_paths = [e["path"] for e in entry["outlinks"]]
            in_paths = [e["path"] for e in entry["inlinks"]]
            assert out_paths == ["C.md"], out_paths
            assert in_paths == ["A.md"], in_paths
            # FileNode attached when the neighbor exists in the store
            assert entry["outlinks"][0]["node"] is not None
            assert entry["outlinks"][0]["node"].path == "C.md"
            assert entry["inlinks"][0]["node"].path == "A.md"
        finally:
            await fs.close()
        print("✓ test_get_neighbors_batch_basic_chain passed")

    _run(run())


def test_get_neighbors_batch_multipath():
    """Querying multiple paths returns one entry per path."""

    async def run():
        items = [
            (_node("A.md", [_link("A.md", "B.md")]), []),
            (_node("B.md", [_link("B.md", "C.md")]), []),
            (_node("C.md", []), []),
        ]
        fs = await _make_store_with_graph(items)
        try:
            result = await get_first_order_neighbors_batch(fs, ["A.md", "B.md", "C.md"])
            assert set(result.keys()) == {"A.md", "B.md", "C.md"}
            assert [e["path"] for e in result["A.md"]["outlinks"]] == ["B.md"]
            assert [e["path"] for e in result["A.md"]["inlinks"]] == []
            assert [e["path"] for e in result["C.md"]["outlinks"]] == []
            assert [e["path"] for e in result["C.md"]["inlinks"]] == ["B.md"]
        finally:
            await fs.close()
        print("✓ test_get_neighbors_batch_multipath passed")

    _run(run())


def test_get_neighbors_max_per_direction_caps():
    """`max_per_direction=2` truncates each direction list."""

    async def run():
        items = [
            (
                _node(
                    "Hub.md",
                    [_link("Hub.md", f"T{i}.md") for i in range(5)],
                ),
                [],
            ),
        ] + [(_node(f"T{i}.md", []), []) for i in range(5)]
        fs = await _make_store_with_graph(items)
        try:
            result = await get_first_order_neighbors_batch(fs, ["Hub.md"], max_per_direction=2)
            entry = result["Hub.md"]
            assert len(entry["outlinks"]) == 2, len(entry["outlinks"])
            # Targets without inlinks: each Tn has Hub as the sole inlink
            for i in range(5):
                t_entry = await get_first_order_neighbors(fs, f"T{i}.md", max_per_direction=2)
                assert [e["path"] for e in t_entry["inlinks"]] == ["Hub.md"]
        finally:
            await fs.close()
        print("✓ test_get_neighbors_max_per_direction_caps passed")

    _run(run())


def test_get_neighbors_anchor_and_predicate_preserved():
    """Edge anchor / predicate flow through into the ``edges`` list, and the
    referenced node's frontmatter is reachable via ``entry['node']``."""

    async def run():
        items = [
            (
                _node(
                    "Source.md",
                    [
                        _link("Source.md", "Target.md", anchor="section1", predicate="ref"),
                        _link("Source.md", "Target.md", anchor="section2", predicate="cites"),
                    ],
                    name="Source Title",
                ),
                [],
            ),
            (_node("Target.md", [], name="Target Title"), []),
        ]
        fs = await _make_store_with_graph(items)
        try:
            # Query Source — outlinks[0] = Target (we should see Target's frontmatter).
            src_result = await get_first_order_neighbors(fs, "Source.md")
            assert len(src_result["outlinks"]) == 1, src_result["outlinks"]
            out_entry = src_result["outlinks"][0]
            assert out_entry["path"] == "Target.md"
            assert out_entry["node"].front_matter.name == "Target Title"
            # Edge grouping: two distinct (predicate, anchor) edges to Target.
            edges = out_entry["edges"]
            assert len(edges) == 2, edges
            assert sorted(e["predicate"] for e in edges) == ["cites", "ref"]
            assert sorted(e["anchor"] for e in edges) == ["section1", "section2"]

            # Query Target — inlinks[0] = Source (we should see Source's frontmatter).
            tgt_result = await get_first_order_neighbors(fs, "Target.md")
            assert len(tgt_result["inlinks"]) == 1, tgt_result["inlinks"]
            in_entry = tgt_result["inlinks"][0]
            assert in_entry["path"] == "Source.md"
            assert in_entry["node"].front_matter.name == "Source Title"
        finally:
            await fs.close()
        print("✓ test_get_neighbors_anchor_and_predicate_preserved passed")

    _run(run())


def test_get_neighbors_no_links_returns_empty_lists():
    """Standalone file with no in/out edges → both lists empty."""

    async def run():
        items = [(_node("Lonely.md", []), [])]
        fs = await _make_store_with_graph(items)
        try:
            result = await get_first_order_neighbors(fs, "Lonely.md")
            assert result == {"outlinks": [], "inlinks": []}
        finally:
            await fs.close()
        print("✓ test_get_neighbors_no_links_returns_empty_lists passed")

    _run(run())


def test_get_neighbors_missing_path_yields_empty_entry():
    """Querying a path not in the store should not crash; returns an empty entry."""

    async def run():
        fs = await _make_store_with_graph([])
        try:
            result = await get_first_order_neighbors(fs, "Ghost.md")
            assert result == {"outlinks": [], "inlinks": []}
        finally:
            await fs.close()
        print("✓ test_get_neighbors_missing_path_yields_empty_entry passed")

    _run(run())


def test_get_neighbors_empty_path_list_returns_empty_dict():
    """`paths=[]` short-circuits to ``{}`` without touching the store."""

    async def run():
        fs = await _make_store_with_graph([])
        try:
            result = await get_first_order_neighbors_batch(fs, [])
            assert result == {}
        finally:
            await fs.close()
        print("✓ test_get_neighbors_empty_path_list_returns_empty_dict passed")

    _run(run())


# ---------------------------------------------------------------------------
# ReadStep rendering helpers
# ---------------------------------------------------------------------------


def test_render_neighbor_block_with_outlinks_and_inlinks():
    """Block contains header, arrows for each direction, and frontmatter line."""
    neighbors = {
        "outlinks": [
            {
                "path": "Out.md",
                "node": _node("Out.md", [], name="Outgoing Title"),
                "edges": [{"predicate": None, "anchor": None}],
            },
        ],
        "inlinks": [
            {
                "path": "In.md",
                "node": _node("In.md", [], description="Incoming desc"),
                "edges": [{"predicate": "ref", "anchor": "h2"}],
            },
        ],
    }
    block = _render_neighbor_block(neighbors)
    assert "outlinks=1" in block
    assert "inlinks=1" in block
    assert "→ Out.md" in block
    assert "← In.md" in block
    assert "Outgoing Title" in block
    assert "Incoming desc" in block
    print("✓ test_render_neighbor_block_with_outlinks_and_inlinks passed")


def test_render_neighbor_block_empty_returns_empty_string():
    """No neighbors in either direction → empty string (caller should skip append)."""
    assert _render_neighbor_block({"outlinks": [], "inlinks": []}) == ""
    print("✓ test_render_neighbor_block_empty_returns_empty_string passed")


def test_neighbors_to_serializable_replaces_filenodes_with_dicts():
    """Serialized form has plain dicts (JSON-friendly), preserves path/edges/node fields."""
    neighbors = {
        "outlinks": [
            {
                "path": "Out.md",
                "node": _node("Out.md", [], name="X"),
                "edges": [{"predicate": "p", "anchor": "a"}],
            },
        ],
        "inlinks": [
            {"path": "Missing.md", "node": None, "edges": []},
        ],
    }
    s = _neighbors_to_serializable(neighbors)
    assert s["outlinks"][0]["path"] == "Out.md"
    assert isinstance(s["outlinks"][0]["node"], dict)
    assert s["outlinks"][0]["node"]["front_matter"]["name"] == "X"
    assert s["outlinks"][0]["edges"] == [{"predicate": "p", "anchor": "a"}]
    assert s["inlinks"][0]["node"] is None
    print("✓ test_neighbors_to_serializable_replaces_filenodes_with_dicts passed")


# ---------------------------------------------------------------------------
# End-to-end: read step with_neighbors via HTTP
# ---------------------------------------------------------------------------


def _seed(working: Path, rel: str, body: str) -> Path:
    target = working / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


async def _wait_for_token(host: str, port: int, token: str, expected_path: str, *, timeout: float = 20.0):
    """Poll search for ``token`` until ``expected_path`` shows up among results."""
    deadline = asyncio.get_event_loop().time() + timeout
    last_paths: list[str] = []
    while asyncio.get_event_loop().time() < deadline:
        r = await call_action("search", host=host, port=port, query=token, limit=20)
        last_paths = [item.get("path", "") for item in (r.get("metadata", {}).get("results") or [])]
        if any(p == expected_path for p in last_paths):
            return
        await asyncio.sleep(0.3)
    raise AssertionError(f"timeout waiting for {expected_path}; last paths={last_paths}")


def test_read_step_neighbor_injection_via_step_kwargs():
    """Drive ReadStep directly with ``with_neighbors=True`` in init kwargs.

    Bypasses HTTP and the job-config layer to prove the injection logic itself
    is correct when properly wired. (The HTTP path is currently broken — see
    ``test_read_step_with_neighbors_via_http_is_currently_inert``.)
    """

    async def run():
        from reme4.components.runtime_context import RuntimeContext
        from reme4.steps.crud.read import ReadStep

        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp) / ".reme"
            memory = working / "memory"
            memory.mkdir(parents=True, exist_ok=True)
            a = _seed(memory, "A.md", "---\nname: Alpha\n---\nhello-A links to [[B]]\n")
            _seed(memory, "B.md", "---\nname: Bravo\n---\nhello-B content\n")

            # Build a file_store with A → B already linked so we don't need
            # to spin up the parser/graph pipeline.
            fs = LocalFileStore(store_name="test_read_inject_store", embedding_model="")
            await fs.start()
            await fs.clear()
            await fs.upsert(
                [
                    (
                        _node(
                            "memory/A.md",
                            [_link("memory/A.md", "memory/B.md")],
                            name="Alpha",
                        ),
                        [],
                    ),
                    (_node("memory/B.md", [], name="Bravo"), []),
                ],
            )
            await fs.rebuild_links()

            try:
                # Construct step with the opt-in kwargs (this is the supported path).
                step = ReadStep(file_store=fs, with_neighbors=True, max_neighbors_per_direction=10)
                # working_path comes from app_context — patch via attribute since we
                # don't want to spin up the full ApplicationContext.
                step.app_context = None  # ensure working_path falls back to Path.cwd()
                old_cwd = os.getcwd()
                os.chdir(working)
                try:
                    ctx = RuntimeContext(path="memory/B.md")
                    resp = await step(ctx)
                finally:
                    os.chdir(old_cwd)  # restore BEFORE tempdir cleanup
            finally:
                await fs.close()

            assert resp.success is True
            answer = resp.answer
            md = resp.metadata
            assert "hello-B" in answer
            assert "link_expansion" in md, md
            in_paths = [e["path"] for e in md["link_expansion"].get("inlinks", [])]
            assert "memory/A.md" in in_paths, in_paths
            assert "Related neighbors" in answer
            assert "← memory/A.md" in answer
            _ = a  # silence unused
        print("✓ test_read_step_neighbor_injection_via_step_kwargs passed")

    _run(run())


def test_read_step_with_neighbors_via_context_kwarg():
    """Pass with_neighbors via RuntimeContext (the HTTP/request path).

    This is the key regression test for the read.py fix: previously the flag
    was only honored via step init kwargs (``self.kwargs``), so HTTP requests
    silently no-op'd. After the fix, ``self.context.get("with_neighbors")``
    takes precedence and the neighbor block is injected as expected.
    """

    async def run():
        from reme4.components.runtime_context import RuntimeContext
        from reme4.steps.crud.read import ReadStep

        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp) / ".reme"
            memory = working / "memory"
            memory.mkdir(parents=True, exist_ok=True)
            _seed(memory, "A.md", "---\nname: Alpha\n---\nhello-A [[B]]\n")
            _seed(memory, "B.md", "---\nname: Bravo\n---\nhello-B content\n")

            fs = LocalFileStore(store_name="test_read_ctx_store", embedding_model="")
            await fs.start()
            await fs.clear()
            await fs.upsert(
                [
                    (
                        _node(
                            "memory/A.md",
                            [_link("memory/A.md", "memory/B.md")],
                            name="Alpha",
                        ),
                        [],
                    ),
                    (_node("memory/B.md", [], name="Bravo"), []),
                ],
            )
            await fs.rebuild_links()

            try:
                # NOTE: with_neighbors is NOT in step init kwargs — only in context.
                step = ReadStep(file_store=fs)
                step.app_context = None
                old_cwd = os.getcwd()
                os.chdir(working)
                try:
                    ctx = RuntimeContext(path="memory/B.md", with_neighbors=True)
                    resp = await step(ctx)
                finally:
                    os.chdir(old_cwd)
            finally:
                await fs.close()

            assert resp.success is True
            md = resp.metadata
            assert "link_expansion" in md, f"context path didn't fire injection; metadata={md}"
            in_paths = [e["path"] for e in md["link_expansion"].get("inlinks", [])]
            assert "memory/A.md" in in_paths, in_paths
            assert "Related neighbors" in resp.answer
            assert "← memory/A.md" in resp.answer
        print("✓ test_read_step_with_neighbors_via_context_kwarg passed")

    _run(run())


def test_read_step_context_overrides_init_kwargs():
    """Context-supplied with_neighbors must override the step's init kwargs.

    Init kwarg says False, context says True → should inject (context wins).
    """

    async def run():
        from reme4.components.runtime_context import RuntimeContext
        from reme4.steps.crud.read import ReadStep

        with tempfile.TemporaryDirectory() as tmp:
            working = Path(tmp) / ".reme"
            memory = working / "memory"
            memory.mkdir(parents=True, exist_ok=True)
            _seed(memory, "A.md", "---\nname: Alpha\n---\nhello [[B]]\n")
            _seed(memory, "B.md", "---\nname: Bravo\n---\nhello B\n")

            fs = LocalFileStore(store_name="test_read_override_store", embedding_model="")
            await fs.start()
            await fs.clear()
            await fs.upsert(
                [
                    (_node("memory/A.md", [_link("memory/A.md", "memory/B.md")], name="Alpha"), []),
                    (_node("memory/B.md", [], name="Bravo"), []),
                ],
            )
            await fs.rebuild_links()

            try:
                # init says with_neighbors=False; context overrides to True.
                step = ReadStep(file_store=fs, with_neighbors=False)
                step.app_context = None
                old_cwd = os.getcwd()
                os.chdir(working)
                try:
                    ctx = RuntimeContext(path="memory/B.md", with_neighbors=True)
                    resp = await step(ctx)
                finally:
                    os.chdir(old_cwd)
            finally:
                await fs.close()

            assert "link_expansion" in resp.metadata, "context value didn't override init kwarg"
        print("✓ test_read_step_context_overrides_init_kwargs passed")

    _run(run())


def test_read_step_with_neighbors_http_omitted_defaults_to_off():
    """When the client doesn't pass with_neighbors, the block is NOT injected."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            memory = working / "memory"
            memory.mkdir(parents=True, exist_ok=True)
            _seed(memory, "A.md", "---\nname: Alpha\n---\nuniqalphax [[B]]\n")
            _seed(memory, "B.md", "---\nname: Bravo\n---\nuniqbravox body\n")

            async with mock_reme_server() as (host, port):
                await _wait_for_token(host, port, "uniqalphax", "memory/A.md")
                await _wait_for_token(host, port, "uniqbravox", "memory/B.md")
                result = await call_action("read", host=host, port=port, path="memory/B.md")

            assert result.get("success") is True
            assert "link_expansion" not in (result.get("metadata") or {})
            assert "Related neighbors" not in result.get("answer", "")
        print("✓ test_read_step_with_neighbors_http_omitted_defaults_to_off passed")

    _run(run())


def test_read_step_default_does_not_inject_neighbors():
    """Without with_neighbors, answer must NOT contain the neighbor block."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            memory = working / "memory"
            memory.mkdir(parents=True, exist_ok=True)
            _seed(memory, "X.md", "---\nname: X\n---\nbody of X\n")

            async with mock_reme_server() as (host, port):
                # No watcher wait needed: with_neighbors=False short-circuits before
                # file_store access. Just read straight from disk.
                result = await call_action("read", host=host, port=port, path="memory/X.md")

            assert result.get("success") is True
            assert "Related neighbors" not in result.get("answer", "")
            assert "link_expansion" not in result.get("metadata", {})
        print("✓ test_read_step_default_does_not_inject_neighbors passed")

    _run(run())


# ---------------------------------------------------------------------------
# Search step regression (post-refactor)
# ---------------------------------------------------------------------------


def test_search_step_expand_links_after_refactor():
    """After refactor, search still surfaces link_expansion in metadata for indexed hits."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            working = Path(tmp) / ".reme"
            memory = working / "memory"
            memory.mkdir(parents=True, exist_ok=True)
            _seed(memory, "Doc.md", "---\nname: My Doc\n---\nuniqsearchzzqq links to [[Other]]\n")
            _seed(memory, "Other.md", "---\nname: Other\n---\nuniqothertoken plain content\n")

            async with mock_reme_server() as (host, port):
                await _wait_for_token(host, port, "uniqsearchzzqq", "memory/Doc.md")

                # expand_links defaults to True. Assert link_expansion is in
                # metadata (shape, not counts — wikilink resolution by the parser/
                # graph is exercised by other tests).
                await call_and_check(
                    "search",
                    host=host,
                    port=port,
                    query="uniqsearchzzqq",
                    limit=5,
                    validator=lambda r: (
                        isinstance(r, dict)
                        and r.get("success") is True
                        and isinstance(r.get("metadata", {}).get("link_expansion"), dict)
                    ),
                )
        print("✓ test_search_step_expand_links_after_refactor passed")

    _run(run())
