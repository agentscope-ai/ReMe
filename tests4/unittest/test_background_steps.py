"""Tests for background steps: ScanChangesStep + WatchChangesStep.

Both steps are subclasses of BaseStep. To exercise them without spinning up the
full ApplicationContext / update_store_index job, we:
  * pass real (started) file_store/file_parser via the step's kwargs (so the
    BaseStep _resolve() machinery returns them);
  * stub run_job() with a small recorder that captures the changes payload.
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from watchfiles import Change

from reme4.components.file_parser import ChunkedFileParser
from reme4.components.file_store import LocalFileStore
from reme4.components.runtime_context import RuntimeContext
from reme4.schema import Response
from reme4.steps.background import ScanChangesStep, WatchChangesStep

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


# ---------------------------------------------------------------------------
# ScanChangesStep
# ---------------------------------------------------------------------------


class _RecorderStep:
    """Mixin: replaces run_job with a recorder that captures the changes payload."""

    recorded: list[dict]
    dispatched: int

    def install_recorder(self):
        """Install a fake run_job that records dispatched 'update_store_index' payloads."""
        self.recorded = []
        self.dispatched = 0

        async def fake_run_job(name: str, **kwargs: Any):
            assert name == "update_store_index"
            self.recorded = kwargs.get("changes") or []
            self.dispatched += 1
            return Response()

        # pylint: disable-next=attribute-defined-outside-init
        self.run_job = fake_run_job  # type: ignore[assignment]


class _RecordingScanChangesStep(ScanChangesStep, _RecorderStep):
    pass


async def _make_scan_step(
    watch_paths: list[str] | str = "vault",
    suffix_filters: list[str] | None = None,
    recursive: bool = True,
    dump_store_index: bool = True,
    dispatch_job: str = "update_store_index",
) -> tuple[_RecordingScanChangesStep, RuntimeContext, LocalFileStore, ChunkedFileParser]:
    fs = LocalFileStore(store_name="test_store", embedding_model="")
    parser = ChunkedFileParser()
    await fs.start()
    await parser.start()
    step = _RecordingScanChangesStep(
        recursive=recursive,
        dump_store_index=dump_store_index,
        dispatch_job=dispatch_job,
        file_store=fs,
        file_parser=parser,
    )
    step.install_recorder()
    context = RuntimeContext(
        watch_paths=watch_paths,
        suffix_filters=suffix_filters or ["md"],
    )
    return step, context, fs, parser


async def _teardown(fs: LocalFileStore, parser: ChunkedFileParser) -> None:
    await parser.close()
    await fs.close()


def test_scan_changes_initial_all_added():
    """First run on a fresh store emits 'added' for every existing file (abs paths)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            # Use Path.cwd() as the basis so we match BaseStep.vault_path on macOS
            # (where /var resolves to /private/var via a symlink).
            cwd = Path.cwd()
            vault = cwd / "vault"
            write_file(vault / "a.md", "alpha")
            write_file(vault / "b.md", "beta")
            step, ctx, fs, parser = await _make_scan_step()
            try:
                resp = await step(ctx)
                counts = resp.metadata["counts"]
                assert counts == {"added": 2, "modified": 0, "deleted": 0}
                assert step.dispatched == 1
                kinds = sorted(item["change"] for item in step.recorded)
                paths = sorted(item["path"] for item in step.recorded)
                assert kinds == ["added", "added"]
                expected = sorted([str(cwd / "vault/a.md"), str(cwd / "vault/b.md")])
                assert paths == expected
            finally:
                await _teardown(fs, parser)
        print("✓ test_scan_changes_initial_all_added passed")

    asyncio.run(run())


def test_scan_changes_no_changes_skips_dispatch():
    """A second run over an unchanged store reports zero counts and does not dispatch."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            vault = cwd / "vault"
            a = write_file(vault / "a.md", "alpha")
            seed_step, ctx, fs, parser = await _make_scan_step()
            try:
                node, chunks = await parser.parse(a)
                await fs.upsert([(node, chunks)])

                resp = await seed_step(ctx)
                counts = resp.metadata["counts"]
                assert counts == {"added": 0, "modified": 0, "deleted": 0}
                assert seed_step.dispatched == 0
            finally:
                await _teardown(fs, parser)
        print("✓ test_scan_changes_no_changes_skips_dispatch passed")

    asyncio.run(run())


def test_scan_changes_detects_modify_and_delete():
    """Second pass distinguishes added/modified/deleted; paths are absolute."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            vault = cwd / "vault"
            a = write_file(vault / "a.md", "alpha")
            b = write_file(vault / "b.md", "beta")
            step, ctx, fs, parser = await _make_scan_step()
            try:
                # Seed via direct parse/upsert.
                for p in (a, b):
                    node, chunks = await parser.parse(p)
                    await fs.upsert([(node, chunks)])

                # Modify a, delete b, add c.
                a.write_text("alpha-v2", encoding="utf-8")
                os.utime(a, (9_999_999_999, 9_999_999_999))
                b.unlink()
                c = write_file(vault / "c.md", "gamma")

                resp = await step(ctx)
                counts = resp.metadata["counts"]
                assert counts == {"added": 1, "modified": 1, "deleted": 1}
                by_kind = {item["change"]: item["path"] for item in step.recorded}
                assert by_kind["added"] == str(c)
                assert by_kind["modified"] == str(a)
                assert by_kind["deleted"] == str(b)
            finally:
                await _teardown(fs, parser)
        print("✓ test_scan_changes_detects_modify_and_delete passed")

    asyncio.run(run())


def test_scan_changes_missing_watch_path_silently_skipped():
    """Non-existent watch_paths entries are dropped silently."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            (Path(tmpdir) / "vault").mkdir()
            step, ctx, fs, parser = await _make_scan_step(watch_paths=["vault", "ghost"])
            try:
                resp = await step(ctx)
                assert resp.metadata["counts"] == {"added": 0, "modified": 0, "deleted": 0}
                assert step.dispatched == 0
            finally:
                await _teardown(fs, parser)
        print("✓ test_scan_changes_missing_watch_path_silently_skipped passed")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# WatchChangesStep
# ---------------------------------------------------------------------------


class _RecordingWatchChangesStep(WatchChangesStep, _RecorderStep):
    pass


def test_watch_changes_requires_stop_event():
    """Missing stop_event in context raises a clear error."""

    async def run():
        step = _RecordingWatchChangesStep()
        step.install_recorder()
        step.context = RuntimeContext(watch_paths=["vault"], suffix_filters=["md"])
        try:
            await step.execute()
        except RuntimeError as e:
            assert "stop_event" in str(e)
        else:
            raise AssertionError("expected RuntimeError")
        print("✓ test_watch_changes_requires_stop_event passed")

    asyncio.run(run())


def test_watch_changes_raises_when_no_valid_paths():
    """With no valid watch_paths, the step raises so the BackgroundJob supervisor can back off."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            step = _RecordingWatchChangesStep()
            step.install_recorder()
            stop = asyncio.Event()
            step.context = RuntimeContext(
                stop_event=stop,
                watch_paths=["ghost"],
                suffix_filters=["md"],
            )

            try:
                await step.execute()
            except RuntimeError as e:
                assert "No valid watch paths" in str(e)
            else:
                raise AssertionError("expected RuntimeError")
            assert step.dispatched == 0
        print("✓ test_watch_changes_raises_when_no_valid_paths passed")

    asyncio.run(run())


def test_watch_changes_filter_only_passes_md():
    """The internal filter pulls suffix_filters from runtime context."""

    step = _RecordingWatchChangesStep()
    step.install_recorder()
    step.context = RuntimeContext(suffix_filters=["md"])
    assert step._filter(Change.added, "/x/foo.md")
    assert not step._filter(Change.added, "/x/foo.txt")
    print("✓ test_watch_changes_filter_only_passes_md passed")


if __name__ == "__main__":
    print("\n=== Background Steps Tests ===")
    # ScanChangesStep
    test_scan_changes_initial_all_added()
    test_scan_changes_no_changes_skips_dispatch()
    test_scan_changes_detects_modify_and_delete()
    test_scan_changes_missing_watch_path_silently_skipped()
    # WatchChangesStep
    test_watch_changes_requires_stop_event()
    test_watch_changes_raises_when_no_valid_paths()
    test_watch_changes_filter_only_passes_md()
    print("\n所有测试通过!")
