"""Tests for LiteFileWatcher (excluding the awatch main loop)."""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from watchfiles import Change

from reme4.components.file_parser import DefaultFileParser
from reme4.components.file_store import LocalFileStore
from reme4.components.file_watcher import LiteFileWatcher

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


async def make_watcher(watch_paths: list[str] | str = "vault", **kwargs) -> LiteFileWatcher:
    """Build a LiteFileWatcher with real (started) file_store/file_parser, but no background loop.

    We replace the bind() Dependency placeholders with concrete instances and start them
    manually, so tests can call update_store / on_* directly without the background task.
    """
    watcher = LiteFileWatcher(watch_paths=watch_paths, **kwargs)
    fs = LocalFileStore(store_name="test_store", embedding_model="")
    parser = DefaultFileParser()
    await fs.start()
    await parser.start()
    watcher.file_store = fs
    watcher.file_parser = parser
    return watcher


async def teardown_watcher(watcher: LiteFileWatcher) -> None:
    """Close the manually-started subcomponents."""
    await watcher.file_parser.close()
    await watcher.file_store.close()


def write_file(path: Path, content: str = "x") -> Path:
    """Create or overwrite a file and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_watch_filter_default_md():
    """Default suffix_filters=['md'] passes .md files and rejects others."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            (Path(tmpdir) / "vault").mkdir()
            watcher = await make_watcher()

            assert watcher.watch_filter(Change.added, "/x/foo.md")
            assert not watcher.watch_filter(Change.added, "/x/foo.txt")
            assert not watcher.watch_filter(Change.added, "/x/foo")

            print("✓ test_watch_filter_default_md passed")

    asyncio.run(run())


def test_watch_filter_custom_suffix():
    """Custom suffix_filters override the default."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            (Path(tmpdir) / "vault").mkdir()
            watcher = await make_watcher(suffix_filters=["txt", ".rst"])

            assert watcher.watch_filter(Change.added, "/x/foo.txt")
            assert watcher.watch_filter(Change.added, "/x/foo.rst")
            assert not watcher.watch_filter(Change.added, "/x/foo.md")

            print("✓ test_watch_filter_custom_suffix passed")

    asyncio.run(run())


def test_watch_filter_no_filter_passes_all():
    """When suffix_filters is empty (set after init), watch_filter passes everything."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            (Path(tmpdir) / "vault").mkdir()
            watcher = await make_watcher()
            # Constructor coerces [] / None → ["md"]; clear it directly to exercise the no-filter branch.
            watcher.suffix_filters = []

            assert watcher.watch_filter(Change.added, "/x/foo")
            assert watcher.watch_filter(Change.added, "/x/foo.md")

            print("✓ test_watch_filter_no_filter_passes_all passed")

    asyncio.run(run())


def test_relative_and_absolute_path_helpers():
    """_get_relative_path strips working_path; _get_absolute_path resolves against it."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            (Path(tmpdir) / "vault").mkdir()
            watcher = await make_watcher()

            # Use watcher.working_path to match the same realpath form (macOS /var ↔ /private/var).
            abs_in = (watcher.working_path / "vault" / "a.md").absolute()
            assert watcher._get_relative_path(abs_in) == "vault/a.md"

            # Path outside working_path → returns absolute
            outside = Path("/opt/elsewhere/x.md").absolute()
            assert watcher._get_relative_path(outside) == str(outside)

            # _get_absolute_path: relative resolves under working_path
            assert watcher._get_absolute_path("vault/a.md") == watcher.working_path / "vault/a.md"
            # absolute stays absolute
            assert watcher._get_absolute_path(str(abs_in)) == abs_in

            print("✓ test_relative_and_absolute_path_helpers passed")

    asyncio.run(run())


def test_scan_existing_files_finds_md_recursive():
    """scan_existing_files returns md files under watch_paths, recursively."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir) / "vault"
            write_file(vault / "a.md", "alpha")
            write_file(vault / "sub" / "b.md", "beta")
            write_file(vault / "ignore.txt", "skip")  # filtered by suffix
            watcher = await make_watcher()

            files = await watcher.scan_existing_files()
            rels = set(files.keys())
            assert "vault/a.md" in rels
            assert "vault/sub/b.md" in rels
            assert "vault/ignore.txt" not in rels

            print("✓ test_scan_existing_files_finds_md_recursive passed")

    asyncio.run(run())


def test_scan_existing_files_non_recursive():
    """recursive=False only scans direct children."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir) / "vault"
            write_file(vault / "a.md", "alpha")
            write_file(vault / "sub" / "b.md", "beta")
            watcher = await make_watcher(recursive=False)

            files = await watcher.scan_existing_files()
            rels = set(files.keys())
            assert "vault/a.md" in rels
            assert "vault/sub/b.md" not in rels

            print("✓ test_scan_existing_files_non_recursive passed")

    asyncio.run(run())


def test_on_added_indexes_files():
    """on_added parses files and writes them into the file_store."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir) / "vault"
            write_file(vault / "a.md", "hello")
            watcher = await make_watcher()
            try:
                await watcher.on_added(["vault/a.md"])
                nodes = await watcher.file_store.file_graph.get_nodes()
                assert {n.path for n in nodes} == {"vault/a.md"}
            finally:
                await teardown_watcher(watcher)
            print("✓ test_on_added_indexes_files passed")

    asyncio.run(run())


def test_on_modified_replaces_node():
    """on_modified re-parses and updates the node entry for an existing path."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir) / "vault"
            file_path = write_file(vault / "a.md", "v1")
            watcher = await make_watcher()
            try:
                await watcher.on_added(["vault/a.md"])
                node_before = (await watcher.file_store.file_graph.get_nodes(["vault/a.md"]))[0]
                # Bump mtime + content
                file_path.write_text("v2 different content", encoding="utf-8")
                os.utime(file_path, (node_before.st_mtime + 10, node_before.st_mtime + 10))

                await watcher.on_modified(["vault/a.md"])
                node_after = (await watcher.file_store.file_graph.get_nodes(["vault/a.md"]))[0]
                assert node_after.st_mtime > node_before.st_mtime
            finally:
                await teardown_watcher(watcher)
            print("✓ test_on_modified_replaces_node passed")

    asyncio.run(run())


def test_on_deleted_removes_node():
    """on_deleted removes the node from the store regardless of file presence."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir) / "vault"
            write_file(vault / "a.md", "hello")
            watcher = await make_watcher()
            try:
                await watcher.on_added(["vault/a.md"])
                assert {n.path for n in await watcher.file_store.file_graph.get_nodes()} == {"vault/a.md"}

                await watcher.on_deleted(["vault/a.md"])
                assert await watcher.file_store.file_graph.get_nodes() == []
            finally:
                await teardown_watcher(watcher)
            print("✓ test_on_deleted_removes_node passed")

    asyncio.run(run())


def test_update_store_initial_add():
    """First update_store run on a fresh store reports all files as added."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir) / "vault"
            write_file(vault / "a.md", "alpha")
            write_file(vault / "b.md", "beta")
            watcher = await make_watcher()
            try:
                counts = await watcher.update_store(dump=False)
                assert counts == {"added": 2, "modified": 0, "deleted": 0}
                paths = {n.path for n in await watcher.file_store.file_graph.get_nodes()}
                assert paths == {"vault/a.md", "vault/b.md"}
            finally:
                await teardown_watcher(watcher)
            print("✓ test_update_store_initial_add passed")

    asyncio.run(run())


def test_update_store_detects_modify_and_delete():
    """update_store distinguishes modified vs deleted vs added on a second pass."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir) / "vault"
            a = write_file(vault / "a.md", "alpha")
            b = write_file(vault / "b.md", "beta")
            watcher = await make_watcher()
            try:
                # Initial sync to seed the store.
                await watcher.update_store(dump=False)

                # Modify a (bump mtime), delete b, add c.
                a.write_text("alpha-v2", encoding="utf-8")
                os.utime(a, (9_999_999_999, 9_999_999_999))
                b.unlink()
                write_file(vault / "c.md", "gamma")

                counts = await watcher.update_store(dump=False)
                assert counts == {"added": 1, "modified": 1, "deleted": 1}
                paths = {n.path for n in await watcher.file_store.file_graph.get_nodes()}
                assert paths == {"vault/a.md", "vault/c.md"}
            finally:
                await teardown_watcher(watcher)
            print("✓ test_update_store_detects_modify_and_delete passed")

    asyncio.run(run())


def test_update_store_no_changes():
    """A second sync over an unchanged tree reports zero counts."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            vault = Path(tmpdir) / "vault"
            write_file(vault / "a.md", "alpha")
            watcher = await make_watcher()
            try:
                await watcher.update_store(dump=False)
                counts = await watcher.update_store(dump=False)
                assert counts == {"added": 0, "modified": 0, "deleted": 0}
            finally:
                await teardown_watcher(watcher)
            print("✓ test_update_store_no_changes passed")

    asyncio.run(run())


def test_missing_watch_path_filtered():
    """watch_paths entries that don't exist are dropped from self.watch_paths."""

    async def run():
        with tempfile.TemporaryDirectory() as tmpdir, temp_chdir(tmpdir):
            (Path(tmpdir) / "vault").mkdir()
            watcher = await make_watcher(watch_paths=["vault", "ghost"])
            assert [p.name for p in watcher.watch_paths] == ["vault"]
            print("✓ test_missing_watch_path_filtered passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== LiteFileWatcher Tests ===")
    test_watch_filter_default_md()
    test_watch_filter_custom_suffix()
    test_watch_filter_no_filter_passes_all()
    test_relative_and_absolute_path_helpers()
    test_scan_existing_files_finds_md_recursive()
    test_scan_existing_files_non_recursive()
    test_on_added_indexes_files()
    test_on_modified_replaces_node()
    test_on_deleted_removes_node()
    test_update_store_initial_add()
    test_update_store_detects_modify_and_delete()
    test_update_store_no_changes()
    test_missing_watch_path_filtered()
    print("\n所有测试通过!")
