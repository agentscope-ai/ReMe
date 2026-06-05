"""Tests for background steps: ScanStoreChangesStep + WatchChangesStep.

Both steps are subclasses of BaseStep. To exercise them without spinning up the
full ApplicationContext, we pass real (started) file_store/file_chunker via the
step's kwargs (so the BaseStep _resolve() machinery returns them).

ScanStoreChangesStep writes its result into ``context["changes"]`` for a downstream
``update_index_step`` to consume; tests assert against that key directly.

The catalog-side sibling (ScanCatalogChangesStep) shares the same diff helper
and is exercised through the dream-loop integration tests; covering it here
would duplicate the file_store-diff assertions without adding signal.
"""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from watchfiles import Change

from reme4.components.file_chunker import DefaultFileChunker
from reme4.components.file_store import LocalFileStore
from reme4.components.runtime_context import RuntimeContext
from reme4.steps import ScanStoreChangesStep, WatchChangesStep

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
# ScanStoreChangesStep
# ---------------------------------------------------------------------------


async def _make_scan_step(
    watch_paths: list[str] | str = "vault",
    suffix_filters: list[str] | None = None,
    recursive: bool = True,
) -> tuple[ScanStoreChangesStep, RuntimeContext, LocalFileStore, DefaultFileChunker]:
    fs = LocalFileStore(name="test_store", embedding_store="")
    parser = DefaultFileChunker()
    await fs.start()
    await parser.start()
    step = ScanStoreChangesStep(
        recursive=recursive,
        file_store=fs,
        file_chunker=parser,
    )
    context = RuntimeContext(
        watch_paths=watch_paths,
        suffix_filters=suffix_filters or ["md"],
    )
    return step, context, fs, parser


async def _teardown(fs: LocalFileStore, parser: DefaultFileChunker) -> None:
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
                changes = ctx["changes"]
                kinds = sorted(item["change"] for item in changes)
                paths = sorted(item["path"] for item in changes)
                assert kinds == ["added", "added"]
                expected = sorted([str(cwd / "vault/a.md"), str(cwd / "vault/b.md")])
                assert paths == expected
            finally:
                await _teardown(fs, parser)
        print("✓ test_scan_changes_initial_all_added passed")

    asyncio.run(run())


def test_scan_changes_no_changes_emits_empty_list():
    """A second run over an unchanged store reports zero counts and empty changes."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            cwd = Path.cwd()
            vault = cwd / "vault"
            a = write_file(vault / "a.md", "alpha")
            step, ctx, fs, parser = await _make_scan_step()
            try:
                node, chunks = await parser.parse(a)
                await fs.upsert([(node, chunks)])

                resp = await step(ctx)
                counts = resp.metadata["counts"]
                assert counts == {"added": 0, "modified": 0, "deleted": 0}
                assert ctx["changes"] == []
            finally:
                await _teardown(fs, parser)
        print("✓ test_scan_changes_no_changes_emits_empty_list passed")

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
                by_kind = {item["change"]: item["path"] for item in ctx["changes"]}
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
                assert ctx["changes"] == []
            finally:
                await _teardown(fs, parser)
        print("✓ test_scan_changes_missing_watch_path_silently_skipped passed")

    asyncio.run(run())


# ---------------------------------------------------------------------------
# WatchChangesStep
# ---------------------------------------------------------------------------


def test_watch_changes_requires_stop_event():
    """Missing stop_event in context raises a clear error."""

    async def run():
        step = WatchChangesStep()
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
            step = WatchChangesStep()
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
        print("✓ test_watch_changes_raises_when_no_valid_paths passed")

    asyncio.run(run())


def test_watch_changes_filter_only_passes_md():
    """The internal filter pulls suffix_filters from runtime context."""

    step = WatchChangesStep()
    step.context = RuntimeContext(suffix_filters=["md"])
    assert step._filter(Change.added, "/x/foo.md")
    assert not step._filter(Change.added, "/x/foo.txt")
    print("✓ test_watch_changes_filter_only_passes_md passed")


if __name__ == "__main__":
    print("\n=== Background Steps Tests ===")
    # ScanStoreChangesStep
    test_scan_changes_initial_all_added()
    test_scan_changes_no_changes_emits_empty_list()
    test_scan_changes_detects_modify_and_delete()
    test_scan_changes_missing_watch_path_silently_skipped()
    # WatchChangesStep
    test_watch_changes_requires_stop_event()
    test_watch_changes_raises_when_no_valid_paths()
    test_watch_changes_filter_only_passes_md()
    print("\n所有测试通过!")
