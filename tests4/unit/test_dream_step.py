"""Tests for DreamStep — the per-change-batch dream + index processor.

DreamStep is the middle step in ``auto_dream_loop``: it consumes
``context.changes`` (filled by ``scan_changes_step`` at startup or by
``watch_changes_step`` at runtime) and runs ``dream_one`` followed by
``UpdateIndexStep`` for each ``added`` / ``modified``; ``deleted``
paths go straight to ``file_store.delete``. Path-shape filtering is
delegated to the watcher (``watch_paths`` + ``suffix_filters``) — this
step trusts whatever lands in ``changes``.

We mock both ``dream_one`` (needs an LLM) and ``UpdateIndexStep``
(needs an ``app_context`` for the file_parser registry). The unit
under test is the fan-out + persistence-gating logic; the inner
steps are exercised in their own files / integration tests.

Covered points:

* Persisting ``file_store`` only on dream success
* Skipped (Phase 1 empty) is still treated as success → still indexed
* Per-file granularity (one failure does not block subsequent files)
* Outside-vault paths are dropped defensively
* ``WatchChangesStep`` exposes awatch ``step`` / ``debounce`` verbatim
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from watchfiles import Change

from reme4.components.file_store import BaseFileStore
from reme4.components.runtime_context import RuntimeContext
from reme4.steps import DreamStep, WatchChangesStep
from reme4.steps.evolve.auto_dream import DreamResult

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class temp_chdir:
    """Context manager that temporarily ``chdir``s into a directory and restores cwd on exit."""

    def __init__(self, path):
        self.path = path
        self.old = None

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)


def _touch(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# DreamStep — fan-out / persistence wiring
#
# Everything below patches ``UpdateIndexStep`` and ``dream_one`` so we
# don't need an LLM or an app_context. The mocked UpdateIndexStep stub
# records each call so we can assert which paths got persisted.
# ---------------------------------------------------------------------------


def _make_step(vault: Path) -> DreamStep:
    """DreamStep with vault_path forced and a fake file_store recording deletes."""
    fake_fs = MagicMock(spec=BaseFileStore)
    fake_fs.delete = AsyncMock()

    class _FixedVault(DreamStep):
        @property
        def vault_path(self):
            return vault

    step = _FixedVault(file_store=fake_fs, persist=True)
    return step


def _patch_update_index_step():
    """Patch UpdateIndexStep at the import site inside dream_step."""
    instances: list[MagicMock] = []

    def _factory(**kwargs):
        m = AsyncMock()
        m.kwargs = kwargs
        instances.append(m)
        return m

    patcher = patch("reme4.steps.evolve.dream_step.UpdateIndexStep", side_effect=_factory)
    return patcher, instances


def test_dream_step_added_modified_persists_on_success():
    """A successful dream on an add/modify must construct + call UpdateIndexStep."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            note = _touch(vault / "daily" / "2026-06-02" / "note.md")
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "added", "path": str(note)}])

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=True, path=rel, summary="ok")

            patcher, instances = _patch_update_index_step()
            with patcher, patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert resp.success
            assert resp.metadata["dreamed"] == 1
            assert len(instances) == 1
            call = instances[0].await_args
            forwarded = call.kwargs["changes"]
            assert forwarded == [{"change": Change.modified, "path": str(note)}]
        print("✓ test_dream_step_added_modified_persists_on_success passed")

    asyncio.run(run())


def test_dream_step_skipped_still_persists():
    """Phase 1 returning empty (skipped=True) is success — still indexed."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            note = _touch(vault / "daily" / "2026-06-02" / "note.md")
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "modified", "path": str(note)}])

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=True, path=rel, skipped=True, summary="empty")

            patcher, instances = _patch_update_index_step()
            with patcher, patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert resp.success
            assert len(instances) == 1
        print("✓ test_dream_step_skipped_still_persists passed")

    asyncio.run(run())


def test_dream_step_failure_does_not_persist():
    """On dream error UpdateIndexStep must not be invoked at all."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            note = _touch(vault / "daily" / "2026-06-02" / "note.md")
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "added", "path": str(note)}])

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=False, path=rel, error="boom")

            patcher, instances = _patch_update_index_step()
            with patcher, patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert not resp.success
            assert resp.metadata["failed"] == 1
            assert not instances
        print("✓ test_dream_step_failure_does_not_persist passed")

    asyncio.run(run())


def test_dream_step_deletes_paths():
    """Deleted paths skip dream entirely and call file_store.delete."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            note_path = vault / "daily" / "2026-06-02" / "note.md"
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "deleted", "path": str(note_path)}])

            patcher, instances = _patch_update_index_step()
            with patcher, patch.object(step, "dream_one") as dream_mock:
                resp = await step(ctx)
                dream_mock.assert_not_called()

            assert resp.success
            assert resp.metadata["deleted"] == 1
            step.file_store.delete.assert_awaited_once_with(["daily/2026-06-02/note.md"])
            assert not instances
        print("✓ test_dream_step_deletes_paths passed")

    asyncio.run(run())


def test_dream_step_drops_paths_outside_vault():
    """Defensive guard: paths the watcher should never have forwarded
    (outside vault root) must be dropped instead of crashing."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as outside:
            vault = Path(tmpdir).resolve()
            external = _touch(Path(outside).resolve() / "stray.md")
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "added", "path": str(external)}])

            patcher, instances = _patch_update_index_step()
            with patcher, patch.object(step, "dream_one") as dream_mock:
                resp = await step(ctx)
                dream_mock.assert_not_called()

            assert resp.success
            assert not instances
        print("✓ test_dream_step_drops_paths_outside_vault passed")

    asyncio.run(run())


def test_dream_step_partial_failure_does_not_block_other_files():
    """File N's failure must not stop file N+1 from being persisted."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            a = _touch(vault / "daily" / "2026-06-02" / "a.md")
            b = _touch(vault / "daily" / "2026-06-02" / "b.md")
            step = _make_step(vault)
            ctx = RuntimeContext(
                changes=[
                    {"change": "added", "path": str(a)},
                    {"change": "added", "path": str(b)},
                ],
            )

            async def _fake_dream(rel, _hint):
                if rel.endswith("a.md"):
                    return DreamResult(used_llm=False, path=rel, error="boom")
                return DreamResult(used_llm=True, path=rel, summary="ok")

            patcher, instances = _patch_update_index_step()
            with patcher, patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert not resp.success
            assert len(instances) == 1
            forwarded = instances[0].await_args.kwargs["changes"]
            assert forwarded == [{"change": Change.modified, "path": str(b)}]
        print("✓ test_dream_step_partial_failure_does_not_block_other_files passed")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# WatchChangesStep — awatch step/debounce passthrough
# ---------------------------------------------------------------------------


def test_watch_changes_defaults_match_awatch_defaults():
    """No-arg construction matches awatch's own defaults."""
    s = WatchChangesStep()
    assert s.step == 50
    assert s.debounce == 2000
    print("✓ test_watch_changes_defaults_match_awatch_defaults passed")


def test_watch_changes_accepts_step_and_debounce_overrides():
    """yaml-supplied step / debounce land verbatim on the instance."""
    s = WatchChangesStep(step=300_000, debounce=600_000)
    assert s.step == 300_000
    assert s.debounce == 600_000
    print("✓ test_watch_changes_accepts_step_and_debounce_overrides passed")


if __name__ == "__main__":
    print("\n=== DreamStep / WatchChanges Tests ===")
    test_dream_step_added_modified_persists_on_success()
    test_dream_step_skipped_still_persists()
    test_dream_step_failure_does_not_persist()
    test_dream_step_deletes_paths()
    test_dream_step_drops_paths_outside_vault()
    test_dream_step_partial_failure_does_not_block_other_files()
    test_watch_changes_defaults_match_awatch_defaults()
    test_watch_changes_accepts_step_and_debounce_overrides()
    print("\n所有测试通过!")
