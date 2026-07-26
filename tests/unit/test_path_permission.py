"""Tests for the request-scoped ``_allowed_paths`` permission mechanism.

The constraint is server-owned: AutoMemoryStep passes it to the agent wrapper
via ``injected_job_kwargs``; the wrapper merges it into every job tool call
(rejecting model-supplied conflicts) and BaseJob places it into the
per-invocation RuntimeContext, where the file I/O steps read it.

Covers:
- ReadStep / EditStep / WriteStep / FrontmatterUpdateStep honoring the scope.
- Exact-file scope (least privilege) vs directory scope, custom daily dirs.
- Fail-closed behavior for invalid injected constraints.
- Wrapper-level injection: merge, conflict rejection, schema hiding
  (AgentScope, Claude Code, Codex serialization).
- AutoMemoryStep injecting ``date`` (create) / ``_allowed_paths`` (update)
  while keeping the original read/edit/write/frontmatter_update tool names.
"""

# pylint: disable=protected-access

import os
import tempfile
from pathlib import Path

import pytest

from reme.components.file_store import LocalFileStore
from reme.steps.file_io.edit import EditStep
from reme.steps.file_io.frontmatter_update import FrontmatterUpdateStep
from reme.steps.file_io.read import ReadStep
from reme.steps.file_io.write import WriteStep


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


def _seed(workspace: Path, rel: str, body: str = "body\n") -> Path:
    target = workspace / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


async def _make_store() -> LocalFileStore:
    store = LocalFileStore(name="t_perm", embedding_store="")
    await store.start()
    return store


async def _run(step_cls, store: LocalFileStore, **kwargs):
    """Run a file I/O step; ``_allowed_paths`` rides in like injected job kwargs."""
    step = step_cls(file_store=store)
    await step(**kwargs)
    return step.context.response


NOTE = "daily/2025-06-01/podcast-habits.md"
SIBLING = "daily/2025-06-01/other-note.md"
OUTSIDE = "topics/roadmap.md"


@pytest.mark.asyncio
async def test_update_scope_allows_exact_note():
    """All four update tools succeed on the exact note_path they are scoped to."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, "---\nname: x\n---\nold text\n")
        store = await _make_store()
        scope = {"_allowed_paths": [NOTE]}

        resp = await _run(ReadStep, store, path=NOTE, **scope)
        assert resp.success is True
        assert "old text" in str(resp.answer)

        resp = await _run(EditStep, store, path=NOTE, old="old text", new="new text", **scope)
        assert resp.success is True

        resp = await _run(FrontmatterUpdateStep, store, path=NOTE, metadata={"name": "renamed"}, **scope)
        assert resp.success is True

        resp = await _run(WriteStep, store, path=NOTE, name="n", description="d", content="rewritten", **scope)
        assert resp.success is True
        assert "rewritten" in (Path(tmp) / NOTE).read_text(encoding="utf-8")
        await store.close()


@pytest.mark.asyncio
async def test_update_scope_rejects_sibling_note_in_same_daily_dir():
    """Exact-file scope denies another note in the same daily directory (least privilege)."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE)
        sibling = _seed(Path(tmp), SIBLING, "---\nname: s\n---\nkeep\n")
        store = await _make_store()
        scope = {"_allowed_paths": [NOTE]}

        for coro in (
            _run(ReadStep, store, path=SIBLING, **scope),
            _run(EditStep, store, path=SIBLING, old="keep", new="gone", **scope),
            _run(WriteStep, store, path=SIBLING, content="overwrite", **scope),
            _run(FrontmatterUpdateStep, store, path=SIBLING, metadata={"name": "hijack"}, **scope),
        ):
            resp = await coro
            assert resp.success is False
            assert "no permission" in str(resp.answer).lower()
        assert "keep" in sibling.read_text(encoding="utf-8")
        await store.close()


@pytest.mark.asyncio
async def test_update_scope_rejects_paths_outside_daily_dir():
    """Exact-file scope denies files elsewhere in the workspace."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE)
        outside = _seed(Path(tmp), OUTSIDE, "roadmap\n")
        store = await _make_store()
        scope = {"_allowed_paths": [NOTE]}

        for coro in (
            _run(ReadStep, store, path=OUTSIDE, **scope),
            _run(EditStep, store, path=OUTSIDE, old="roadmap", new="x", **scope),
            _run(WriteStep, store, path=OUTSIDE, content="x", **scope),
            _run(FrontmatterUpdateStep, store, path=OUTSIDE, metadata={"k": "v"}, **scope),
        ):
            resp = await coro
            assert resp.success is False
            assert "no permission" in str(resp.answer).lower()
        assert outside.read_text(encoding="utf-8") == "roadmap\n"
        await store.close()


@pytest.mark.asyncio
async def test_scope_follows_markdown_suffix_gating():
    """A model path without ``.md`` gates to the same file and stays in scope."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, "gated\n")
        store = await _make_store()

        resp = await _run(ReadStep, store, path=NOTE.removesuffix(".md"), _allowed_paths=[NOTE])
        assert resp.success is True
        assert "gated" in str(resp.answer)
        await store.close()


@pytest.mark.asyncio
async def test_custom_daily_dir_scope_needs_no_config_jobs():
    """A customized daily_dir (e.g. journal/) works because the scope is the note path itself."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        note = "journal/2025-06-01/trip.md"
        _seed(Path(tmp), note, "trip\n")
        _seed(Path(tmp), "journal/2025-06-01/other.md")
        store = await _make_store()
        scope = {"_allowed_paths": [note]}

        resp = await _run(ReadStep, store, path=note, **scope)
        assert resp.success is True

        resp = await _run(WriteStep, store, path="journal/2025-06-01/other.md", content="x", **scope)
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_directory_scope_allows_nested_paths():
    """A directory entry acts as a prefix scope with path-component boundaries."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "daily/2025-06-01/deep/nested.md", "nested\n")
        _seed(Path(tmp), "daily-report/leak.md", "leak\n")
        store = await _make_store()
        scope = {"_allowed_paths": ["daily"]}

        resp = await _run(ReadStep, store, path="daily/2025-06-01/deep/nested.md", **scope)
        assert resp.success is True

        resp = await _run(ReadStep, store, path="daily-report/leak.md", **scope)
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_nonexistent_allowed_paths_are_skipped():
    """Missing injected entries do not grant access or invalidate valid entries."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "journal/kept.md", "kept\n")
        _seed(Path(tmp), "outside.md", "outside\n")
        store = await _make_store()

        scope = {"_allowed_paths": ["missing.md", "journal"]}
        resp = await _run(ReadStep, store, path="journal/kept.md", **scope)
        assert resp.success is True

        resp = await _run(ReadStep, store, path="outside.md", **scope)
        assert resp.success is False

        resp = await _run(ReadStep, store, path="journal/kept.md", _allowed_paths=["missing.md"])
        assert resp.success is False
        assert "no permission" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_invalid_injected_constraints_fail_closed():
    """Empty lists, escaping entries, and non-list values all deny access."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), NOTE, "target\n")
        store = await _make_store()

        for bad_scope in ([], ["../escape.md"], ["/etc/passwd"], 42, {"path": NOTE}):
            resp = await _run(ReadStep, store, path=NOTE, _allowed_paths=bad_scope)
            assert resp.success is False, f"scope {bad_scope!r} should fail closed"
            assert "no permission" in str(resp.answer).lower()
        await store.close()


@pytest.mark.asyncio
async def test_no_constraint_allows_all():
    """Without ``_allowed_paths`` the steps impose no additional restriction."""
    with tempfile.TemporaryDirectory() as tmp, temp_chdir(tmp):
        _seed(Path(tmp), "anywhere/x.md", "free\n")
        store = await _make_store()

        resp = await _run(ReadStep, store, path="anywhere/x.md")
        assert resp.success is True

        resp = await _run(WriteStep, store, path="anywhere/new.md", content="ok")
        assert resp.success is True
        await store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
