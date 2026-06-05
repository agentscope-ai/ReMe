"""Tests for background steps: scan/watch/dispatch steps.

Both scan steps are subclasses of BaseStep. To exercise them without spinning up
the full ApplicationContext, we pass real (started) file_store/file_chunker via
the step's kwargs (so the BaseStep _resolve() machinery returns them).

ScanStoreChangesStep writes its result into ``context["changes"]`` for a
downstream ``update_index_step`` to consume; tests assert against that key.
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from watchfiles import Change

from reme4.components.file_chunker import DefaultFileChunker
from reme4.components.file_store import LocalFileStore
from reme4.components.runtime_context import RuntimeContext
from reme4.steps import ForeachDispatchStep, LogChangesStep, ScanStoreChangesStep, WatchChangesStep
from reme4.steps.index._watch_rules import WatchRule, build_watch_rules, collect_existing, match_file

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


def write_file(path: Path, content: str = "x") -> Path:
    """Create parent dirs and write `content` to `path`; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_app_context(vault_path: Path, daily_dir="daily", digest_dir="digest", resource_dir="resource"):
    """Create a mock app_context with app_config pointing to the given vault."""
    ctx = MagicMock()
    ctx.app_config.vault_dir = str(vault_path)
    ctx.app_config.daily_dir = daily_dir
    ctx.app_config.digest_dir = digest_dir
    ctx.app_config.resource_dir = resource_dir
    return ctx


# ---------------------------------------------------------------------------
# _watch_rules module tests
# ---------------------------------------------------------------------------


def test_build_watch_rules_basic():
    """Build rules from watch_dirs and watch_suffixes."""
    app_config = MagicMock()
    app_config.daily_dir = "daily"
    app_config.digest_dir = "digest"
    app_config.resource_dir = "resource"
    vault = Path("/fake/vault")

    rules = build_watch_rules(app_config, vault, watch_dirs=["daily_dir", "digest_dir"], watch_suffixes=["md"])
    assert len(rules) == 2
    assert rules[0].path == vault / "daily"
    assert rules[0].suffixes == ["md"]
    assert rules[1].path == vault / "digest"
    print("✓ test_build_watch_rules_basic passed")


def test_build_watch_rules_multiple_suffixes():
    """Multiple suffixes are forwarded to each rule."""
    app_config = MagicMock()
    app_config.daily_dir = "daily"
    app_config.resource_dir = "resource"
    vault = Path("/fake/vault")

    rules = build_watch_rules(
        app_config,
        vault,
        watch_dirs=["daily_dir", "resource_dir"],
        watch_suffixes=["md", "jsonl"],
    )
    assert len(rules) == 2
    assert rules[0].suffixes == ["md", "jsonl"]
    assert rules[1].suffixes == ["md", "jsonl"]
    print("✓ test_build_watch_rules_multiple_suffixes passed")


def test_build_watch_rules_fallback_literal():
    """Unknown field names are used as literal directory names."""
    app_config = MagicMock(spec=[])  # no attributes
    vault = Path("/fake/vault")
    rules = build_watch_rules(app_config, vault, watch_dirs=["custom_dir"], watch_suffixes=["txt"])
    assert rules[0].path == vault / "custom_dir"
    print("✓ test_build_watch_rules_fallback_literal passed")


def test_match_file_suffix():
    """match_file accepts files matching suffix under rule path."""
    rules = [WatchRule(path=Path("/vault/daily"), suffixes=["md"])]
    assert match_file("/vault/daily/2026-01-01.md", rules)
    assert match_file("/vault/daily/sub/note.md", rules)
    assert not match_file("/vault/daily/file.txt", rules)
    assert not match_file("/vault/other/file.md", rules)
    print("✓ test_match_file_suffix passed")


def test_match_file_no_suffix_filter():
    """Empty suffixes list means all files match."""
    rules = [WatchRule(path=Path("/vault/resource"), suffixes=[])]
    assert match_file("/vault/resource/anything.xyz", rules)
    assert match_file("/vault/resource/sub/deep.pdf", rules)
    assert not match_file("/vault/other/file.md", rules)
    print("✓ test_match_file_no_suffix_filter passed")


def test_collect_existing_filters():
    """collect_existing applies suffix rules correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        daily = vault / "daily"
        resource = vault / "resource"
        write_file(daily / "note.md")
        write_file(daily / "ignore.txt")
        write_file(resource / "data.json")
        write_file(resource / "binary.png")

        rules = [
            WatchRule(path=daily, suffixes=["md"]),
            WatchRule(path=resource, suffixes=["json"]),
        ]
        result = collect_existing(rules, recursive=True)
        paths = set(result.keys())
        assert str((daily / "note.md").absolute()) in paths
        assert str((daily / "ignore.txt").absolute()) not in paths
        assert str((resource / "data.json").absolute()) in paths
        assert str((resource / "binary.png").absolute()) not in paths
    print("✓ test_collect_existing_filters passed")


# ---------------------------------------------------------------------------
# ScanStoreChangesStep
# ---------------------------------------------------------------------------


async def _make_scan_step(vault_path: Path, watch_dirs=None, watch_suffixes=None, recursive=True):
    fs = LocalFileStore(name="test_store", embedding_store="")
    chunker = DefaultFileChunker()
    await fs.start()
    await chunker.start()
    app_ctx = _make_app_context(vault_path)
    step = ScanStoreChangesStep(recursive=recursive, file_store=fs, file_chunker=chunker, app_context=app_ctx)
    context = RuntimeContext(
        watch_dirs=watch_dirs or ["daily_dir", "digest_dir"],
        watch_suffixes=watch_suffixes or ["md"],
    )
    return step, context, fs, chunker


async def _teardown(fs: LocalFileStore, chunker: DefaultFileChunker) -> None:
    await chunker.close()
    await fs.close()


def test_scan_changes_initial_all_added():
    """First run on a fresh store emits 'added' for every existing file."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            write_file(cwd / "daily" / "a.md", "alpha")
            write_file(cwd / "daily" / "b.md", "beta")
            (cwd / "digest").mkdir(parents=True, exist_ok=True)

            step, ctx, fs, chunker = await _make_scan_step(cwd)
            try:
                resp = await step(ctx)
                counts = resp.metadata["counts"]
                assert counts == {"added": 2, "modified": 0, "deleted": 0}
                assert len(ctx["changes"]) == 2
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_initial_all_added passed")

    asyncio.run(run())


def test_scan_changes_no_changes():
    """Second run over an unchanged store reports zero counts."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            a = write_file(cwd / "daily" / "a.md", "alpha")
            (cwd / "digest").mkdir(parents=True, exist_ok=True)

            step, ctx, fs, chunker = await _make_scan_step(cwd)
            try:
                node, chunks = await chunker.chunk(a)
                await fs.upsert([(node, chunks)])
                resp = await step(ctx)
                assert resp.metadata["counts"] == {"added": 0, "modified": 0, "deleted": 0}
                assert ctx["changes"] == []
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_no_changes passed")

    asyncio.run(run())


def test_scan_changes_detect_modify_delete():
    """Second pass distinguishes added/modified/deleted."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            a = write_file(cwd / "daily" / "a.md", "alpha")
            b = write_file(cwd / "daily" / "b.md", "beta")
            (cwd / "digest").mkdir(parents=True, exist_ok=True)

            step, ctx, fs, chunker = await _make_scan_step(cwd)
            try:
                for p in (a, b):
                    node, chunks = await chunker.chunk(p)
                    await fs.upsert([(node, chunks)])
                a.write_text("alpha-v2", encoding="utf-8")
                os.utime(a, (9_999_999_999, 9_999_999_999))
                b.unlink()
                write_file(cwd / "daily" / "c.md", "gamma")

                resp = await step(ctx)
                counts = resp.metadata["counts"]
                assert counts == {"added": 1, "modified": 1, "deleted": 1}
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_detect_modify_delete passed")

    asyncio.run(run())


def test_scan_changes_missing_dir_skipped():
    """Non-existent watch_dirs entries are dropped silently."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            (cwd / "daily").mkdir()
            # digest dir missing
            step, ctx, fs, chunker = await _make_scan_step(cwd)
            try:
                resp = await step(ctx)
                assert resp.metadata["counts"] == {"added": 0, "modified": 0, "deleted": 0}
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_missing_dir_skipped passed")

    asyncio.run(run())


def test_scan_changes_resource_dir():
    """Scanning resource_dir with multiple suffixes works."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            resource = cwd / "resource"
            write_file(resource / "data.json", "{}")
            write_file(resource / "note.md", "# Note")
            write_file(resource / "image.png", "binary")

            step, ctx, fs, chunker = await _make_scan_step(
                cwd,
                watch_dirs=["resource_dir"],
                watch_suffixes=["md", "json"],
            )
            try:
                resp = await step(ctx)
                assert resp.metadata["counts"]["added"] == 2
            finally:
                await _teardown(fs, chunker)
        print("✓ test_scan_changes_resource_dir passed")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# WatchChangesStep
# ---------------------------------------------------------------------------


def test_watch_changes_requires_stop_event():
    """Missing stop_event in context raises a clear error."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            (cwd / "daily").mkdir()
            app_ctx = _make_app_context(cwd)
            step = WatchChangesStep(app_context=app_ctx)
            step.context = RuntimeContext(watch_dirs=["daily_dir"], watch_suffixes=["md"])
            try:
                await step.execute()
            except RuntimeError as e:
                assert "stop_event" in str(e)
            else:
                raise AssertionError("expected RuntimeError")
        print("✓ test_watch_changes_requires_stop_event passed")

    asyncio.run(run())


def test_watch_changes_raises_no_valid_paths():
    """With no valid watch_paths, the step raises."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            step = WatchChangesStep(app_context=app_ctx)
            stop = asyncio.Event()
            step.context = RuntimeContext(stop_event=stop, watch_dirs=["daily_dir"], watch_suffixes=["md"])
            try:
                await step.execute()
            except RuntimeError as e:
                assert "No valid watch paths" in str(e)
            else:
                raise AssertionError("expected RuntimeError")
        print("✓ test_watch_changes_raises_no_valid_paths passed")

    asyncio.run(run())


def test_watch_changes_filter_matches_rules():
    """The internal filter uses watch rules from context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        (vault / "daily").mkdir()
        (vault / "digest").mkdir()
        (vault / "resource").mkdir()
        app_ctx = _make_app_context(vault)

        step = WatchChangesStep(app_context=app_ctx)
        step.context = RuntimeContext(watch_dirs=["daily_dir", "digest_dir"], watch_suffixes=["md"])
        step._rules = step._get_watch_rules()

        assert step._filter(Change.added, str(vault / "daily/foo.md"))
        assert step._filter(Change.added, str(vault / "digest/bar.md"))
        assert not step._filter(Change.added, str(vault / "daily/foo.txt"))
        assert not step._filter(Change.added, str(vault / "resource/file.md"))

    print("✓ test_watch_changes_filter_matches_rules passed")


def test_watch_changes_dispatch_steps_list():
    """dispatch_steps config properly merges dispatch_step and dispatch_steps."""
    step1 = WatchChangesStep(dispatch_step="update_index_step")
    assert step1.dispatch_steps == ["update_index_step"]

    step2 = WatchChangesStep(dispatch_steps=["update_catalog_step", "foreach_dispatch_step"])
    assert step2.dispatch_steps == ["update_catalog_step", "foreach_dispatch_step"]

    step3 = WatchChangesStep(dispatch_step="a", dispatch_steps=["b", "c"])
    assert step3.dispatch_steps == ["b", "c"]  # dispatch_steps takes priority

    print("✓ test_watch_changes_dispatch_steps_list passed")


# ---------------------------------------------------------------------------
# ForeachDispatchStep
# ---------------------------------------------------------------------------


def test_foreach_dispatch_no_job():
    """Without dispatch_job, step logs warning and returns success."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            step = ForeachDispatchStep(app_context=app_ctx)
            ctx = RuntimeContext(changes=[{"change": "added", "path": "/x/y.md"}])
            resp = await step(ctx)
            assert resp.success is True
            assert resp.metadata.get("dispatched") is None  # skipped early
        print("✓ test_foreach_dispatch_no_job passed")

    asyncio.run(run())


def test_foreach_dispatch_calls_job():
    """ForeachDispatchStep calls run_job for each change."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            step = ForeachDispatchStep(app_context=app_ctx)
            changes = [
                {"change": "added", "path": str(cwd / "resource/2026-01-01/file.md")},
                {"change": "modified", "path": str(cwd / "resource/2026-01-01/data.json")},
            ]
            ctx = RuntimeContext(changes=changes, dispatch_job="auto_resource")

            mock_job = AsyncMock()
            app_ctx.jobs = {"auto_resource": mock_job}
            resp = await step(ctx)
            assert resp.success is True
            assert resp.metadata["dispatched"] == 2
            assert mock_job.call_count == 2
            # Verify vault-relative paths were passed
            calls = mock_job.call_args_list
            assert calls[0].kwargs["file_path"] == "resource/2026-01-01/file.md"
            assert calls[0].kwargs["change"] == "added"
            assert calls[1].kwargs["file_path"] == "resource/2026-01-01/data.json"
            assert calls[1].kwargs["change"] == "modified"
        print("✓ test_foreach_dispatch_calls_job passed")

    asyncio.run(run())


def test_foreach_dispatch_handles_error():
    """ForeachDispatchStep continues on job failure."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            app_ctx = _make_app_context(cwd)
            step = ForeachDispatchStep(app_context=app_ctx)
            changes = [
                {"change": "added", "path": str(cwd / "resource/a.md")},
                {"change": "added", "path": str(cwd / "resource/b.md")},
            ]
            ctx = RuntimeContext(changes=changes, dispatch_job="failing_job")

            mock_job = AsyncMock(side_effect=RuntimeError("boom"))
            app_ctx.jobs = {"failing_job": mock_job}
            resp = await step(ctx)
            assert resp.success is True  # still succeeds
            assert mock_job.call_count == 2  # tried both
        print("✓ test_foreach_dispatch_handles_error passed")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# LogChangesStep
# ---------------------------------------------------------------------------


def test_log_changes_step():
    """LogChangesStep logs and reports count."""

    async def run():
        step = LogChangesStep()
        changes = [
            {"change": "added", "path": "/vault/daily/note.md"},
            {"change": "deleted", "path": "/vault/daily/old.md"},
        ]
        ctx = RuntimeContext(changes=changes)
        resp = await step(ctx)
        assert resp.success is True
        assert resp.metadata["count"] == 2
        print("✓ test_log_changes_step passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== Background Steps Tests ===")
    # _watch_rules
    test_build_watch_rules_basic()
    test_build_watch_rules_multiple_suffixes()
    test_build_watch_rules_fallback_literal()
    test_match_file_suffix()
    test_match_file_no_suffix_filter()
    test_collect_existing_filters()
    # ScanStoreChangesStep
    test_scan_changes_initial_all_added()
    test_scan_changes_no_changes()
    test_scan_changes_detect_modify_delete()
    test_scan_changes_missing_dir_skipped()
    test_scan_changes_resource_dir()
    # WatchChangesStep
    test_watch_changes_requires_stop_event()
    test_watch_changes_raises_no_valid_paths()
    test_watch_changes_filter_matches_rules()
    test_watch_changes_dispatch_steps_list()
    # ForeachDispatchStep
    test_foreach_dispatch_no_job()
    test_foreach_dispatch_calls_job()
    test_foreach_dispatch_handles_error()
    # LogChangesStep
    test_log_changes_step()
    print("\n所有测试通过!")
