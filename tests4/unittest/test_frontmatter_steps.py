"""Tests for frontmatter steps — RUD on the frontmatter slice of markdown files.

Covers ``frontmatter_read_step`` / ``frontmatter_update_step`` / ``frontmatter_delete_step``.
Non-markdown targets get ``error="not markdown"`` and the call is a no-op.
"""

# pylint: disable=protected-access

import asyncio
import json
import os
import tempfile
import warnings
from pathlib import Path

from reme4.components.file_parser.linked_file_parser import _extract_links
from reme4.components.file_store import LocalFileStore
from reme4.schema import FileNode
from reme4.steps.frontmatter import (
    delete as fm_delete,
    read as fm_read,
    update as fm_update,
)

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


async def _make_store(files: dict[str, str] | None = None) -> LocalFileStore:
    """LocalFileStore seeded with files on disk + registered in the graph."""
    store = LocalFileStore(store_name="t", embedding_model="")
    await store.start()
    nodes: list[FileNode] = []
    for rel, content in (files or {}).items():
        abs_path = Path.cwd() / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        nodes.append(
            FileNode(
                path=rel,
                st_mtime=abs_path.stat().st_mtime,
                links=_extract_links(content, rel),
            ),
        )
    if nodes:
        await store.file_graph.upsert_nodes(nodes)
    return store


def _answer(step) -> dict:
    return json.loads(step.context.response.answer)


# -- frontmatter_read_step ----------------------------------------------------


def test_frontmatter_read_returns_frontmatter():
    """frontmatter_read_step returns the parsed YAML frontmatter dict."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {"topics/n.md": "---\nname: T\ndescription: d\n---\nbody"},
            )
            step = fm_read.FrontmatterReadStep(file_store=store)
            await step(path="topics/n.md")
            payload = _answer(step)
            assert payload["exists"] is True
            assert payload["frontmatter"] == {"name": "T", "description": "d"}
            await store.close()
        print("✓ test_frontmatter_read_returns_frontmatter passed")

    asyncio.run(run())


# -- frontmatter_update_step --------------------------------------------------


def test_frontmatter_update_merges_metadata_into_frontmatter():
    """frontmatter_update_step merges the ``metadata`` dict into frontmatter."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {"topics/n.md": "---\nname: T\n---\nbody"},
            )
            step = fm_update.FrontmatterUpdateStep(file_store=store)
            await step(
                path="topics/n.md",
                metadata={"description": "d", "extra": "x"},
            )
            payload = _answer(step)
            assert "error" not in payload
            assert payload["updated"] == {"description": "d", "extra": "x"}
            new_text = (Path(tmp) / "topics/n.md").read_text(encoding="utf-8")
            assert "description: d" in new_text
            assert "extra: x" in new_text
            await store.close()
        print("✓ test_frontmatter_update_merges_metadata_into_frontmatter passed")

    asyncio.run(run())


def test_frontmatter_update_rejects_non_markdown():
    """frontmatter_update_step on a .txt file returns error='not markdown'."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store({"materials/foo.txt": "x"})
            step = fm_update.FrontmatterUpdateStep(file_store=store)
            await step(path="materials/foo.txt", metadata={"a": "1"})
            payload = _answer(step)
            assert payload["error"] == "not markdown"
            await store.close()
        print("✓ test_frontmatter_update_rejects_non_markdown passed")

    asyncio.run(run())


def test_frontmatter_update_empty_metadata_is_error():
    """An empty metadata dict yields error='no fields to update'."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {"topics/n.md": "---\nname: T\n---\nbody"},
            )
            step = fm_update.FrontmatterUpdateStep(file_store=store)
            await step(path="topics/n.md", metadata={})
            payload = _answer(step)
            assert payload["error"] == "no fields to update"
            await store.close()
        print("✓ test_frontmatter_update_empty_metadata_is_error passed")

    asyncio.run(run())


# -- frontmatter_delete_step --------------------------------------------------


def test_frontmatter_delete_drops_keys():
    """frontmatter_delete_step removes listed keys; reports deleted + missing."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
            store = await _make_store(
                {"topics/n.md": "---\nname: T\ndescription: d\n---\nbody"},
            )
            step = fm_delete.FrontmatterDeleteStep(file_store=store)
            await step(path="topics/n.md", keys=["description", "nope"])
            payload = _answer(step)
            assert payload["deleted"] == ["description"]
            assert payload["missing"] == ["nope"]
            assert payload["frontmatter"] == {"name": "T"}
            await store.close()
        print("✓ test_frontmatter_delete_drops_keys passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== frontmatter step tests ===")
    test_frontmatter_read_returns_frontmatter()
    test_frontmatter_update_merges_metadata_into_frontmatter()
    test_frontmatter_update_rejects_non_markdown()
    test_frontmatter_update_empty_metadata_is_error()
    test_frontmatter_delete_drops_keys()
    print("\n所有测试通过!")
