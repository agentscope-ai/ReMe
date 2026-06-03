"""Tests for ``ReadStep`` with ``with_neighbors=True`` opt-in injection.

The opt-in re-uses ``reme4.utils.link_expansion.{expand_links,
render_expansion_lines}`` (already covered by ``test_link_expansion.py``),
so these tests focus on the *integration* layer:

* off by default — no extra answer suffix, no ``link_expansion`` metadata
* on + has neighbors — answer ends with a ``Related neighbors`` block,
  metadata carries the raw expansion dict
* on + no neighbors — no block appended, no metadata key
* on + non-markdown — falls through (no neighbor injection)
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from reme4.components.file_store import LocalFileStore
from reme4.schema import FileFrontMatter, FileNode
from reme4.steps.file_io import read as crud_read
from reme4.utils.wikilink_handler import WikilinkHandler

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


def _run(coro):
    asyncio.run(coro)


async def _store_with(files: dict[str, dict]) -> LocalFileStore:
    """LocalFileStore seeded with files + parsed wikilinks + optional frontmatter."""
    store = LocalFileStore(name="t_read_neighbors", embedding_store="")
    await store.start()
    nodes: list[FileNode] = []
    root = Path.cwd()
    for rel, spec in files.items():
        body = spec["body"]
        abs_path = root / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(body, encoding="utf-8")
        fm = FileFrontMatter(
            name=spec.get("name", ""),
            description=spec.get("description", ""),
        )
        nodes.append(
            FileNode(
                path=rel,
                st_mtime=abs_path.stat().st_mtime,
                links=WikilinkHandler.extract_links(body, rel),
                front_matter=fm,
            ),
        )
    if nodes:
        await store.file_graph.upsert_nodes(nodes)
    return store


async def _read(store: LocalFileStore, *, step_kwargs: dict | None = None, **call_kwargs):
    """Run a ReadStep against ``store``; ``step_kwargs`` go to step init (kwargs/attrs)."""
    step = crud_read.ReadStep(file_store=store, **(step_kwargs or {}))
    await step(**call_kwargs)
    return step.context.response


# -- off (default) -----------------------------------------------------------


def test_read_without_neighbors_default():
    """``with_neighbors`` defaults to off — no block, no metadata key."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _store_with(
                {
                    "A.md": {"body": "See [[B.md]] for details.", "name": "A Doc"},
                    "B.md": {"body": "End node.", "name": "B Doc"},
                },
            )
            resp = await _read(store, path="A.md")
            assert resp.success is True
            assert "Related neighbors" not in str(resp.answer)
            assert "link_expansion" not in resp.metadata
            await store.close()
        print("✓ test_read_without_neighbors_default passed")

    _run(run())


# -- on + has neighbors ------------------------------------------------------


def test_read_with_neighbors_injects_block_and_metadata():
    """Block is appended; metadata carries the raw expansion dict."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _store_with(
                {
                    "A.md": {"body": "See [[B.md]] for details.", "name": "A Doc", "description": "alpha"},
                    "B.md": {"body": "End node.", "name": "B Doc", "description": "beta"},
                },
            )
            resp = await _read(store, step_kwargs={"with_neighbors": True}, path="A.md")
            assert resp.success is True
            answer = str(resp.answer)
            assert "Related neighbors" in answer
            assert "outlinks=1" in answer
            assert "inlinks=0" in answer
            assert "B.md" in answer
            assert 'name="B Doc"' in answer

            expansion = resp.metadata.get("link_expansion")
            assert expansion is not None
            assert "A.md" in expansion
            assert expansion["A.md"]["outlinks"][0]["path"] == "B.md"
            assert expansion["A.md"]["outlinks"][0]["meta"] == {"name": "B Doc", "description": "beta"}
            await store.close()
        print("✓ test_read_with_neighbors_injects_block_and_metadata passed")

    _run(run())


# -- on + zero neighbors -----------------------------------------------------


def test_read_with_neighbors_no_links_no_block():
    """File with no out/in links → no block appended, no metadata key."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _store_with({"lonely.md": {"body": "no links here", "name": "Lonely"}})
            resp = await _read(store, step_kwargs={"with_neighbors": True}, path="lonely.md")
            assert resp.success is True
            assert "Related neighbors" not in str(resp.answer)
            assert "link_expansion" not in resp.metadata
            await store.close()
        print("✓ test_read_with_neighbors_no_links_no_block passed")

    _run(run())


# -- on + non-md -------------------------------------------------------------


def test_read_with_neighbors_non_md_falls_through():
    """Non-md target: neighbor injection is skipped even when ``with_neighbors=True``."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            (Path(tmp) / "notes.txt").write_text("plain text body", encoding="utf-8")
            store = LocalFileStore(name="t_read_neighbors_nonmd", embedding_store="")
            await store.start()
            resp = await _read(store, step_kwargs={"with_neighbors": True}, path="notes.txt")
            assert resp.success is True
            assert "Related neighbors" not in str(resp.answer)
            assert "link_expansion" not in resp.metadata
            await store.close()
        print("✓ test_read_with_neighbors_non_md_falls_through passed")

    _run(run())
