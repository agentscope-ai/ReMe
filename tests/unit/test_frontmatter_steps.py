"""Tests for the frontmatter-only CRUD steps (``frontmatter_read`` /
``frontmatter_update`` / ``frontmatter_delete``).

Covers the suffix-less path contract shared with ``read`` / ``write`` /
``edit``: a bare path with no suffix auto-appends ``.md``. All three
siblings must agree on path handling, disclose the substitution via
``resolved_path`` in the response metadata, and report the actually
probed path in not-found errors instead of the caller's raw input.
"""

# pylint: disable=protected-access

import os
import tempfile
from pathlib import Path

import pytest

from reme.components.file_store import LocalFileStore
from reme.steps.file_io.frontmatter_delete import FrontmatterDeleteStep
from reme.steps.file_io.frontmatter_read import FrontmatterReadStep
from reme.steps.file_io.frontmatter_update import FrontmatterUpdateStep


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


def _seed(workspace: Path, rel: str, body: str) -> Path:
    target = workspace / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


async def _make_store() -> LocalFileStore:
    store = LocalFileStore(name="t_fm", embedding_store="")
    await store.start()
    return store


async def _run(step_cls, store: LocalFileStore, **kwargs):
    step = step_cls(file_store=store)
    await step(**kwargs)
    return step.context.response


NOTE = "notes/post.md"
BODY = "---\nname: n\ntags:\n- a\n---\nbody\n"


@pytest.mark.asyncio
async def test_read_no_suffix_autoappends_md():
    """frontmatter_read on a suffix-less path resolves to the ``.md`` file."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path="notes/post")
        assert resp.success is True
        assert resp.metadata["frontmatter"] == {"name": "n", "tags": ["a"]}
        assert resp.metadata["path"] == "notes/post"
        assert resp.metadata["resolved_path"] == "notes/post.md"
        await store.close()


@pytest.mark.asyncio
async def test_update_no_suffix_autoappends_md():
    """frontmatter_update on a suffix-less path resolves to the ``.md`` file."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        note = _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        resp = await _run(FrontmatterUpdateStep, store, path="notes/post", metadata={"name": "renamed"})
        assert resp.success is True
        assert resp.metadata["updated"] == {"name": "renamed"}
        assert resp.metadata["resolved_path"] == "notes/post.md"
        assert "name: renamed" in note.read_text(encoding="utf-8")
        await store.close()


@pytest.mark.asyncio
async def test_delete_no_suffix_autoappends_md():
    """frontmatter_delete on a suffix-less path resolves to the ``.md`` file."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        note = _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        resp = await _run(FrontmatterDeleteStep, store, path="notes/post", keys=["tags"])
        assert resp.success is True
        assert resp.metadata["deleted"] == ["tags"]
        assert resp.metadata["missing"] == []
        assert resp.metadata["resolved_path"] == "notes/post.md"
        assert "tags" not in note.read_text(encoding="utf-8")
        await store.close()


@pytest.mark.asyncio
async def test_family_consistent_on_suffixless_path():
    """read / update / delete all succeed on the same suffix-less path."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()

        resp = await _run(FrontmatterReadStep, store, path="notes/post")
        assert resp.success is True

        resp = await _run(FrontmatterUpdateStep, store, path="notes/post", metadata={"status": "done"})
        assert resp.success is True

        resp = await _run(FrontmatterDeleteStep, store, path="notes/post", keys=["status"])
        assert resp.success is True
        assert resp.metadata["deleted"] == ["status"]
        await store.close()


@pytest.mark.asyncio
async def test_read_missing_reports_probed_path():
    """not-found errors surface the probed ``.md`` path, not the raw input."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path="digest/missing")
        assert resp.success is False
        assert resp.metadata["exists"] is False
        assert resp.metadata["path"] == "digest/missing"
        assert resp.metadata["resolved_path"] == "digest/missing.md"
        assert "digest/missing.md" in str(resp.answer)
        await store.close()


@pytest.mark.asyncio
async def test_update_missing_reports_probed_path():
    """frontmatter_update not-found errors mention the probed ``.md`` path."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        resp = await _run(FrontmatterUpdateStep, store, path="digest/missing", metadata={"x": 1})
        assert resp.success is False
        assert resp.metadata["error"] == "digest/missing.md not found"
        assert resp.metadata["resolved_path"] == "digest/missing.md"
        assert "digest/missing.md" in str(resp.answer)
        await store.close()


@pytest.mark.asyncio
async def test_delete_missing_reports_probed_path():
    """frontmatter_delete not-found errors mention the probed ``.md`` path."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        store = await _make_store()
        resp = await _run(FrontmatterDeleteStep, store, path="digest/missing", keys=["x"])
        assert resp.success is False
        assert resp.metadata["error"] == "digest/missing.md not found"
        assert resp.metadata["resolved_path"] == "digest/missing.md"
        assert "digest/missing.md" in str(resp.answer)
        await store.close()


@pytest.mark.asyncio
async def test_non_md_rejected_without_resolved_path():
    """Non-markdown targets are rejected by all three siblings; no substitution."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "notes/data.txt", "plain\n")
        store = await _make_store()

        for step_cls, kwargs in (
            (FrontmatterReadStep, {}),
            (FrontmatterUpdateStep, {"metadata": {"x": 1}}),
            (FrontmatterDeleteStep, {"keys": ["x"]}),
        ):
            resp = await _run(step_cls, store, path="notes/data.txt", **kwargs)
            assert resp.success is False
            assert "not markdown" in str(resp.answer).lower()
            assert "resolved_path" not in resp.metadata
        await store.close()


@pytest.mark.asyncio
async def test_explicit_md_path_has_no_resolved_path():
    """Explicit ``.md`` paths need no substitution, so no ``resolved_path``."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, BODY)
        store = await _make_store()
        resp = await _run(FrontmatterReadStep, store, path=NOTE)
        assert resp.success is True
        assert "resolved_path" not in resp.metadata
        await store.close()
