"""Tests for AutoDreamStep — daily-tick + file_catalog dedup.

AutoDreamStep walks ``daily/<today>.md`` + ``daily/<today>/**`` and
diffs the result against ``file_catalog`` (same shape as
``scan_catalog_changes_step``):

* on-disk path missing from catalog          → dream
* on-disk mtime != catalog mtime             → dream (modified)
* on-disk mtime == catalog mtime             → skip (unchanged)
* catalog entry under today's prefix missing → drop from catalog (deleted)

Successful dreams (and Phase 1 vacuous skips) upsert the current
``st_mtime`` so the next tick re-dreams only what actually changed.
Failures leave the catalog untouched.

We mock ``dream_one`` (needs an LLM) and inject a fake ``file_catalog``
recording every get / upsert / delete / dump.
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
from reme4.schema import FileNode
from reme4.steps import AutoDreamStep
from reme4.steps.evolve.dream import DreamResult

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


def _make_step(vault: Path, today: str, existing_nodes: list[FileNode] | None = None) -> AutoDreamStep:
    """AutoDreamStep with vault forced, daily_dir='daily', and a fake catalog."""
    fake_catalog = MagicMock(spec=BaseFileCatalog)
    fake_catalog.get_nodes = AsyncMock(return_value=list(existing_nodes or []))
    fake_catalog.upsert = AsyncMock()
    fake_catalog.delete = AsyncMock()
    fake_catalog.dump = AsyncMock()

    class _Fixed(AutoDreamStep):
        @property
        def vault_path(self):
            return vault

        def _vault_dir(self):
            return vault

        def _now(self):
            import datetime

            return datetime.datetime.fromisoformat(f"{today}T00:00:00")

    step = _Fixed(file_catalog=fake_catalog, persist=True)
    cfg = MagicMock()
    cfg.daily_dir = "daily"
    cfg.resource_dir = ""
    step.app_context = MagicMock()
    step.app_context.app_config = cfg
    return step


def test_scans_date_md_and_date_folder():
    """Both ``daily/<today>.md`` and files under ``daily/<today>/`` are picked up;
    date.md is dreamed first so the day-index leads."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            today = "2026-06-04"
            _touch(vault / "daily" / f"{today}.md")
            _touch(vault / "daily" / today / "session-a.md")
            _touch(vault / "daily" / today / "session-b.md")
            step = _make_step(vault, today)
            ctx = RuntimeContext()

            seen: list[str] = []

            async def _fake_dream(rel, _hint):
                seen.append(rel)
                return DreamResult(used_llm=True, path=rel, summary="ok")

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert resp.success
            assert resp.metadata["files_scanned"] == 3
            assert resp.metadata["files_dreamed"] == 3
            assert seen[0] == f"daily/{today}.md"
            assert seen[1:] == [f"daily/{today}/session-a.md", f"daily/{today}/session-b.md"]
        print("✓ test_scans_date_md_and_date_folder passed")

    asyncio.run(run())


def test_resource_dir_is_not_scanned():
    """resource/<today>/ files are NOT picked up by AutoDreamStep anymore."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            today = "2026-06-04"
            _touch(vault / "daily" / f"{today}.md")
            _touch(vault / "resource" / today / "spec.pdf")
            step = _make_step(vault, today)
            step.app_context.app_config.resource_dir = "resource"
            ctx = RuntimeContext()

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=True, path=rel, summary="ok")

            with patch.object(step, "dream_one", side_effect=_fake_dream) as dream_mock:
                await step(ctx)

            paths = [c.args[0] for c in dream_mock.call_args_list]
            assert paths == [f"daily/{today}.md"]
        print("✓ test_resource_dir_is_not_scanned passed")

    asyncio.run(run())


def test_unchanged_files_skipped_via_catalog_mtime():
    """A file whose catalog mtime matches on-disk mtime is NOT dreamed."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            today = "2026-06-04"
            note = _touch(vault / "daily" / today / "note.md")
            mtime = note.stat().st_mtime
            existing = [FileNode(path=f"daily/{today}/note.md", st_mtime=mtime)]
            step = _make_step(vault, today, existing_nodes=existing)
            ctx = RuntimeContext()

            with patch.object(step, "dream_one") as dream_mock:
                resp = await step(ctx)
                dream_mock.assert_not_called()

            assert resp.success
            assert resp.metadata["files_unchanged"] == 1
            assert resp.metadata["files_dreamed"] == 0
            step.file_catalog.upsert.assert_not_awaited()
            step.file_catalog.dump.assert_not_awaited()
        print("✓ test_unchanged_files_skipped_via_catalog_mtime passed")

    asyncio.run(run())


def test_changed_file_dreamed_and_catalog_updated():
    """Stale catalog mtime → file is dreamed + catalog upserts new mtime."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            today = "2026-06-04"
            note = _touch(vault / "daily" / today / "note.md")
            mtime = note.stat().st_mtime
            stale = mtime - 999.0
            existing = [FileNode(path=f"daily/{today}/note.md", st_mtime=stale)]
            step = _make_step(vault, today, existing_nodes=existing)
            ctx = RuntimeContext()

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=True, path=rel, summary="ok")

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert resp.success
            assert resp.metadata["files_dreamed"] == 1
            assert resp.metadata["files_unchanged"] == 0
            step.file_catalog.upsert.assert_awaited_once()
            (nodes,), _ = step.file_catalog.upsert.call_args
            assert len(nodes) == 1
            assert nodes[0].path == f"daily/{today}/note.md"
            assert nodes[0].st_mtime == mtime  # post-dream mtime, not stale
            step.file_catalog.dump.assert_awaited_once()
        print("✓ test_changed_file_dreamed_and_catalog_updated passed")

    asyncio.run(run())


def test_deleted_file_dropped_from_catalog():
    """Catalog entry under today's prefix with no on-disk file → catalog.delete."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            today = "2026-06-04"
            # No on-disk files for today; catalog has a stale entry for today.
            existing = [FileNode(path=f"daily/{today}/gone.md", st_mtime=123.0)]
            step = _make_step(vault, today, existing_nodes=existing)
            ctx = RuntimeContext()

            with patch.object(step, "dream_one") as dream_mock:
                resp = await step(ctx)
                dream_mock.assert_not_called()

            assert resp.success
            assert resp.metadata["files_deleted"] == 1
            step.file_catalog.delete.assert_awaited_once_with([f"daily/{today}/gone.md"])
            step.file_catalog.upsert.assert_not_awaited()
            step.file_catalog.dump.assert_awaited_once()
        print("✓ test_deleted_file_dropped_from_catalog passed")

    asyncio.run(run())


def test_other_days_catalog_entries_untouched():
    """Catalog entries OUTSIDE today's prefix must not be dropped, even if
    they don't appear in today's on-disk scan."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            today = "2026-06-04"
            yesterday = "2026-06-03"
            # Today: nothing on disk. Catalog has yesterday's entry.
            existing = [FileNode(path=f"daily/{yesterday}/note.md", st_mtime=99.0)]
            step = _make_step(vault, today, existing_nodes=existing)
            ctx = RuntimeContext()

            resp = await step(ctx)

            assert resp.success
            assert resp.metadata["files_scanned"] == 0
            assert resp.metadata["files_deleted"] == 0
            step.file_catalog.delete.assert_not_awaited()
            step.file_catalog.upsert.assert_not_awaited()
        print("✓ test_other_days_catalog_entries_untouched passed")

    asyncio.run(run())


def test_failure_does_not_upsert():
    """A dream error must leave the catalog untouched so the next tick retries."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            today = "2026-06-04"
            _touch(vault / "daily" / today / "note.md")
            step = _make_step(vault, today)
            ctx = RuntimeContext()

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=False, path=rel, error="boom")

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert not resp.success
            assert resp.metadata["files_failed"] == 1
            step.file_catalog.upsert.assert_not_awaited()
            step.file_catalog.dump.assert_not_awaited()
        print("✓ test_failure_does_not_upsert passed")

    asyncio.run(run())


def test_phase1_empty_still_upserts():
    """Phase 1 skipped (no abstractions) is still success — still catalogued so
    the next tick doesn't redo Phase 1 unnecessarily."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            today = "2026-06-04"
            note = _touch(vault / "daily" / today / "note.md")
            mtime = note.stat().st_mtime
            step = _make_step(vault, today)
            ctx = RuntimeContext()

            async def _fake_dream(rel, _hint):
                return DreamResult(used_llm=True, path=rel, skipped=True, summary="empty")

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert resp.success
            assert resp.metadata["files_skipped"] == 1
            step.file_catalog.upsert.assert_awaited_once()
            (nodes,), _ = step.file_catalog.upsert.call_args
            assert nodes[0].st_mtime == mtime
        print("✓ test_phase1_empty_still_upserts passed")

    asyncio.run(run())


def test_partial_failure_does_not_block_other_files():
    """One file's failure must not stop other files from being dreamed + catalogued."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir).resolve()
            today = "2026-06-04"
            _touch(vault / "daily" / today / "a.md")
            _touch(vault / "daily" / today / "b.md")
            step = _make_step(vault, today)
            ctx = RuntimeContext()

            async def _fake_dream(rel, _hint):
                if rel.endswith("a.md"):
                    return DreamResult(used_llm=False, path=rel, error="boom")
                return DreamResult(used_llm=True, path=rel, summary="ok")

            with patch.object(step, "dream_one", side_effect=_fake_dream):
                resp = await step(ctx)

            assert not resp.success
            assert resp.metadata["files_dreamed"] == 1
            assert resp.metadata["files_failed"] == 1
            step.file_catalog.upsert.assert_awaited_once()
            (nodes,), _ = step.file_catalog.upsert.call_args
            assert [n.path for n in nodes] == [f"daily/{today}/b.md"]
        print("✓ test_partial_failure_does_not_block_other_files passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== AutoDreamStep Tests ===")
    test_scans_date_md_and_date_folder()
    test_resource_dir_is_not_scanned()
    test_unchanged_files_skipped_via_catalog_mtime()
    test_changed_file_dreamed_and_catalog_updated()
    test_deleted_file_dropped_from_catalog()
    test_other_days_catalog_entries_untouched()
    test_failure_does_not_upsert()
    test_phase1_empty_still_upserts()
    test_partial_failure_does_not_block_other_files()
    print("\n所有测试通过!")
