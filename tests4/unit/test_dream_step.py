"""Tests for DreamStep — the per-change-batch dream + catalog processor.

DreamStep is the middle step in ``auto_dream_loop``: it consumes
``context.changes`` (filled by ``scan_changes_step`` at startup or by
``watch_changes_step`` at runtime) and runs ``dream_one`` for each
``added`` / ``modified``; ``deleted`` paths go straight to
``file_catalog.delete``.

DreamStep does **not** touch ``file_store`` — index writes are owned
by ``update_store_index_loop`` so the two background loops don't race
on the same on-disk artefact. Instead, it tracks "which paths have
been dreamed at which mtime" in the bound ``file_catalog``: a
successful (or vacuously-skipped) dream upserts a ``FileNode``
keyed by vault-relative path; a failure leaves the catalog untouched
so the next ``scan_changes_step source=file_catalog`` re-reports the
file and we retry.

We mock ``dream_one`` (needs an LLM) and inject a fake ``file_catalog``
so the unit under test is the fan-out + state-recording logic; the
inner step + the real catalog are exercised in their own files /
integration tests.

Covered points:

* Catalog upsert only on dream success (with FileNode carrying the
  current ``st_mtime``)
* Skipped (Phase 1 empty) is still success → still catalogued
* Per-file granularity (one failure does not block subsequent files)
* ``deleted`` changes call ``file_catalog.delete`` and skip dream entirely
* Outside-vault paths are dropped defensively
* ``persist=True`` triggers a single ``file_catalog.dump`` per batch
* ``WatchChangesStep`` exposes awatch ``step`` / ``debounce`` verbatim
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from reme4.components.file_catalog import BaseFileCatalog
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
# DreamStep — fan-out / catalog wiring
#
# We patch ``dream_one`` so we don't need an LLM, and inject a fake
# ``file_catalog`` (AsyncMock) recording every upsert / delete / dump.
# ---------------------------------------------------------------------------


def _make_step(vault: Path, persist: bool = True) -> DreamStep:
    """DreamStep with vault_path forced and a fake file_catalog recording calls."""
    fake_catalog = MagicMock(spec=BaseFileCatalog)
    fake_catalog.upsert = AsyncMock()
    fake_catalog.delete = AsyncMock()
    fake_catalog.dump = AsyncMock()

    class _FixedVault(DreamStep):
        @property
        def vault_path(self):
            return vault

    return _FixedVault(file_catalog=fake_catalog, persist=persist)


def test_dream_step_added_modified_upserts_on_success():
    """A successful dream must upsert a FileNode (path + st_mtime) into file_catalog."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            note = _touch(vault / "daily" / "2026-06-02" / "note.md")
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "added", "path": str(note)}])

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=True, path=rel, summary="ok")

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert resp.success
            assert resp.metadata["dreamed"] == 1
            step.file_catalog.upsert.assert_awaited_once()
            (nodes,), _ = step.file_catalog.upsert.call_args
            assert len(nodes) == 1
            assert nodes[0].path == "daily/2026-06-02/note.md"
            assert nodes[0].st_mtime == note.stat().st_mtime
            step.file_catalog.dump.assert_awaited_once()
        print("✓ test_dream_step_added_modified_upserts_on_success passed")

    asyncio.run(run())


def test_dream_step_skipped_still_upserts():
    """Phase 1 returning empty (skipped=True) is success — still catalogued."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            note = _touch(vault / "daily" / "2026-06-02" / "note.md")
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "modified", "path": str(note)}])

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=True, path=rel, skipped=True, summary="empty")

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert resp.success
            step.file_catalog.upsert.assert_awaited_once()
            step.file_catalog.dump.assert_awaited_once()
        print("✓ test_dream_step_skipped_still_upserts passed")

    asyncio.run(run())


def test_dream_step_failure_does_not_upsert():
    """On dream error the catalog must not be touched (next scan will re-report)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            note = _touch(vault / "daily" / "2026-06-02" / "note.md")
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "added", "path": str(note)}])

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=False, path=rel, error="boom")

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert not resp.success
            assert resp.metadata["failed"] == 1
            step.file_catalog.upsert.assert_not_awaited()
            step.file_catalog.dump.assert_not_awaited()
        print("✓ test_dream_step_failure_does_not_upsert passed")

    asyncio.run(run())


def test_dream_step_deletes_paths():
    """Deleted paths skip dream entirely and call file_catalog.delete."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            note_path = vault / "daily" / "2026-06-02" / "note.md"
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "deleted", "path": str(note_path)}])

            with patch.object(step, "dream_one") as dream_mock:
                resp = await step(ctx)
                dream_mock.assert_not_called()

            assert resp.success
            assert resp.metadata["deleted"] == 1
            step.file_catalog.delete.assert_awaited_once_with(["daily/2026-06-02/note.md"])
            step.file_catalog.upsert.assert_not_awaited()
            step.file_catalog.dump.assert_awaited_once()
        print("✓ test_dream_step_deletes_paths passed")

    asyncio.run(run())


def test_dream_step_drops_paths_outside_vault():
    """Defensive guard: paths outside vault root are dropped silently."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as outside:
            vault = Path(tmpdir).resolve()
            external = _touch(Path(outside).resolve() / "stray.md")
            step = _make_step(vault)
            ctx = RuntimeContext(changes=[{"change": "added", "path": str(external)}])

            with patch.object(step, "dream_one") as dream_mock:
                resp = await step(ctx)
                dream_mock.assert_not_called()

            assert resp.success
            step.file_catalog.upsert.assert_not_awaited()
            step.file_catalog.delete.assert_not_awaited()
            step.file_catalog.dump.assert_not_awaited()
        print("✓ test_dream_step_drops_paths_outside_vault passed")

    asyncio.run(run())


def test_dream_step_partial_failure_does_not_block_other_files():
    """File N's failure must not stop file N+1 from being catalogued."""

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

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert not resp.success
            assert resp.metadata["dreamed"] == 1
            assert resp.metadata["failed"] == 1
            step.file_catalog.upsert.assert_awaited_once()
            (nodes,), _ = step.file_catalog.upsert.call_args
            assert [n.path for n in nodes] == ["daily/2026-06-02/b.md"]
        print("✓ test_dream_step_partial_failure_does_not_block_other_files passed")

    asyncio.run(run())


def test_dream_step_persist_false_skips_dump():
    """persist=False → upsert still happens but no dump (caller controls flush)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            note = _touch(vault / "daily" / "2026-06-02" / "note.md")
            step = _make_step(vault, persist=False)
            ctx = RuntimeContext(changes=[{"change": "added", "path": str(note)}])

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=True, path=rel, summary="ok")

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert resp.success
            step.file_catalog.upsert.assert_awaited_once()
            step.file_catalog.dump.assert_not_awaited()
        print("✓ test_dream_step_persist_false_skips_dump passed")

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
    test_dream_step_added_modified_upserts_on_success()
    test_dream_step_skipped_still_upserts()
    test_dream_step_failure_does_not_upsert()
    test_dream_step_deletes_paths()
    test_dream_step_drops_paths_outside_vault()
    test_dream_step_partial_failure_does_not_block_other_files()
    test_dream_step_persist_false_skips_dump()
    test_watch_changes_defaults_match_awatch_defaults()
    test_watch_changes_accepts_step_and_debounce_overrides()
    print("\n所有测试通过!")
