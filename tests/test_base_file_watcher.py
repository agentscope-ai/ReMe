"""
Async unit tests for BaseFileWatcher covering:
- Existing paths and files monitoring
- Non-existent paths handling
- File suffix filtering
- Start/stop lifecycle
- Callback functionality
- scan_on_start feature

Usage:
    pytest tests/test_base_file_watcher.py -v
    pytest tests/test_base_file_watcher.py -v -k "test_existing"
"""

# pylint: disable=redefined-outer-name,protected-access,unused-argument

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from watchfiles import Change

from reme.core.file_watcher.base_file_watcher import BaseFileWatcher


# ==================== Fixtures ====================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_files(temp_dir: Path):
    """Create temporary test files."""
    files = {}

    # Create .txt files
    for i in range(3):
        file_path = temp_dir / f"test_file_{i}.txt"
        file_path.write_text(f"Content of test file {i}")
        files[f"txt_{i}"] = file_path

    # Create .py files
    for i in range(2):
        file_path = temp_dir / f"test_script_{i}.py"
        file_path.write_text(f"# Python script {i}\nprint('hello')")
        files[f"py_{i}"] = file_path

    # Create .md file
    md_file = temp_dir / "readme.md"
    md_file.write_text("# README")
    files["md_0"] = md_file

    yield files


@pytest.fixture
def temp_nested_dir(temp_dir: Path):
    """Create nested directory structure."""
    # Create subdirectories
    sub1 = temp_dir / "subdir1"
    sub1.mkdir()
    sub2 = temp_dir / "subdir2"
    sub2.mkdir()
    nested = sub1 / "nested"
    nested.mkdir()

    # Create files in subdirectories
    (sub1 / "file1.txt").write_text("subdir1 file")
    (sub2 / "file2.txt").write_text("subdir2 file")
    (nested / "nested_file.txt").write_text("nested file")

    yield temp_dir


# ==================== Test Existing Paths ====================


class TestExistingPaths:
    """Tests for existing paths and files."""

    @pytest.mark.asyncio
    async def test_init_with_single_existing_path(self, temp_dir: Path):
        """Test initialization with a single existing path."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        assert watcher.watch_paths == [str(temp_dir)]
        assert watcher.recursive is False
        assert watcher.is_running() is False

    @pytest.mark.asyncio
    async def test_init_with_multiple_existing_paths(self, temp_dir: Path):
        """Test initialization with multiple existing paths."""
        sub1 = temp_dir / "dir1"
        sub2 = temp_dir / "dir2"
        sub1.mkdir()
        sub2.mkdir()

        watcher = BaseFileWatcher(watch_paths=[str(sub1), str(sub2)])

        assert len(watcher.watch_paths) == 2
        assert str(sub1) in watcher.watch_paths
        assert str(sub2) in watcher.watch_paths

    @pytest.mark.asyncio
    async def test_init_with_existing_file(self, temp_files):
        """Test initialization with existing file path."""
        file_path = temp_files["txt_0"]
        watcher = BaseFileWatcher(watch_paths=str(file_path))

        assert watcher.watch_paths == [str(file_path)]

    @pytest.mark.asyncio
    async def test_start_with_existing_path(self, temp_dir: Path):
        """Test starting watcher with existing path."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        await watcher.start()
        assert watcher.is_running() is True

        await watcher.close()
        assert watcher.is_running() is False

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, temp_dir: Path):
        """Test watcher start/stop lifecycle."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        # Start
        await watcher.start()
        assert watcher.is_running() is True
        assert watcher._watch_task is not None

        # Stop
        await watcher.close()
        assert watcher.is_running() is False

        # Restart
        await watcher.start()
        assert watcher.is_running() is True

        await watcher.close()

    @pytest.mark.asyncio
    async def test_multiple_start_calls(self, temp_dir: Path):
        """Test that multiple start calls don't create multiple tasks."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        await watcher.start()
        task1 = watcher._watch_task

        await watcher.start()  # Second call should be ignored
        task2 = watcher._watch_task

        assert task1 is task2
        await watcher.close()

    @pytest.mark.asyncio
    async def test_multiple_close_calls(self, temp_dir: Path):
        """Test that multiple close calls are safe."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        await watcher.start()
        await watcher.close()
        await watcher.close()  # Second call should be safe

        assert watcher.is_running() is False


# ==================== Test Non-Existent Paths ====================


class TestNonExistentPaths:
    """Tests for non-existent paths handling."""

    @pytest.mark.asyncio
    async def test_init_with_nonexistent_path(self):
        """Test initialization with non-existent path."""
        nonexistent = "/path/that/does/not/exist"
        watcher = BaseFileWatcher(watch_paths=nonexistent)

        assert watcher.watch_paths == [nonexistent]

    @pytest.mark.asyncio
    async def test_start_with_nonexistent_path(self):
        """Test starting watcher with non-existent path (should handle gracefully)."""
        nonexistent = "/path/that/does/not/exist"
        watcher = BaseFileWatcher(watch_paths=nonexistent)

        await watcher.start()
        assert watcher.is_running() is True

        # Give it a moment to enter the watch loop and detect the missing path
        await asyncio.sleep(0.1)

        await watcher.close()
        assert watcher.is_running() is False

    @pytest.mark.asyncio
    async def test_mixed_existing_and_nonexistent_paths(self, temp_dir: Path):
        """Test with mix of existing and non-existent paths."""
        nonexistent = "/path/that/does/not/exist"
        watcher = BaseFileWatcher(watch_paths=[str(temp_dir), nonexistent])

        await watcher.start()
        assert watcher.is_running() is True

        # Give it time to filter paths
        await asyncio.sleep(0.1)

        await watcher.close()

    @pytest.mark.asyncio
    async def test_all_paths_nonexistent(self):
        """Test when all paths are non-existent."""
        watcher = BaseFileWatcher(
            watch_paths=["/nonexistent1", "/nonexistent2"],
        )

        await watcher.start()
        assert watcher.is_running() is True

        # Wait for retry logic
        await asyncio.sleep(0.2)

        await watcher.close()

    @pytest.mark.asyncio
    async def test_empty_watch_paths(self):
        """Test with empty watch paths list."""
        watcher = BaseFileWatcher(watch_paths=[])

        await watcher.start()
        assert watcher.is_running() is True

        await asyncio.sleep(0.1)
        await watcher.close()


# ==================== Test File Filtering ====================


class TestFileFiltering:
    """Tests for file suffix filtering."""

    @pytest.mark.asyncio
    async def test_watch_filter_no_filters(self, temp_files):
        """Test watch_filter with no suffix filters (should match all)."""
        watcher = BaseFileWatcher(watch_paths="/tmp")

        assert watcher.watch_filter(Change.added, "test.txt") is True
        assert watcher.watch_filter(Change.added, "test.py") is True
        assert watcher.watch_filter(Change.added, "test.md") is True
        assert watcher.watch_filter(Change.added, "noextension") is True

    @pytest.mark.asyncio
    async def test_watch_filter_with_txt_suffix(self):
        """Test watch_filter with .txt suffix filter."""
        watcher = BaseFileWatcher(watch_paths="/tmp", suffix_filters=[".txt"])

        assert watcher.watch_filter(Change.added, "test.txt") is True
        assert watcher.watch_filter(Change.added, "test.py") is False
        assert watcher.watch_filter(Change.added, "file.txt.bak") is False

    @pytest.mark.asyncio
    async def test_watch_filter_with_multiple_suffixes(self):
        """Test watch_filter with multiple suffix filters."""
        watcher = BaseFileWatcher(
            watch_paths="/tmp",
            suffix_filters=[".txt", ".py", ".md"],
        )

        assert watcher.watch_filter(Change.added, "test.txt") is True
        assert watcher.watch_filter(Change.added, "script.py") is True
        assert watcher.watch_filter(Change.added, "readme.md") is True
        assert watcher.watch_filter(Change.added, "config.json") is False

    @pytest.mark.asyncio
    async def test_watch_filter_suffix_without_dot(self):
        """Test watch_filter handles suffixes without leading dot."""
        watcher = BaseFileWatcher(
            watch_paths="/tmp",
            suffix_filters=["txt", "py"],  # Without dots
        )

        assert watcher.watch_filter(Change.added, "test.txt") is True
        assert watcher.watch_filter(Change.added, "script.py") is True

    @pytest.mark.asyncio
    async def test_watch_filter_all_change_types(self):
        """Test watch_filter works with all Change types."""
        watcher = BaseFileWatcher(
            watch_paths="/tmp",
            suffix_filters=[".txt"],
        )

        # All change types should work with filter
        assert watcher.watch_filter(Change.added, "test.txt") is True
        assert watcher.watch_filter(Change.modified, "test.txt") is True
        assert watcher.watch_filter(Change.deleted, "test.txt") is True


# ==================== Test Callback Functionality ====================


class TestCallbackFunctionality:
    """Tests for callback functionality."""

    @pytest.mark.asyncio
    async def test_sync_callback(self, temp_dir: Path):
        """Test synchronous callback function."""
        callback_called = []

        def sync_callback(changes):
            callback_called.append(changes)

        watcher = BaseFileWatcher(
            watch_paths=str(temp_dir),
            callback=sync_callback,
        )

        # Simulate changes
        test_changes = {(Change.added, str(temp_dir / "test.txt"))}
        await watcher.on_changes(test_changes)

        assert len(callback_called) == 1
        assert callback_called[0] == test_changes

    @pytest.mark.asyncio
    async def test_async_callback(self, temp_dir: Path):
        """Test asynchronous callback function."""
        callback_called = []

        async def async_callback(changes):
            callback_called.append(changes)

        watcher = BaseFileWatcher(
            watch_paths=str(temp_dir),
            callback=async_callback,
        )

        # Simulate changes
        test_changes = {(Change.modified, str(temp_dir / "test.txt"))}
        await watcher.on_changes(test_changes)

        assert len(callback_called) == 1
        assert callback_called[0] == test_changes

    @pytest.mark.asyncio
    async def test_no_callback_uses_internal_handler(self, temp_dir: Path):
        """Test that without callback, internal _on_changes is used."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        # Mock internal _on_changes
        watcher._on_changes = AsyncMock()

        test_changes = {(Change.added, str(temp_dir / "test.txt"))}
        await watcher.on_changes(test_changes)

        watcher._on_changes.assert_called_once_with(test_changes)


# ==================== Test Scan on Start ====================


class TestScanOnStart:
    """Tests for scan_on_start feature."""

    @pytest.mark.asyncio
    async def test_scan_on_start_false(self, temp_files, temp_dir: Path):
        """Test that scan_on_start=False doesn't scan existing files."""
        callback_called = []

        async def callback(changes):
            callback_called.append(changes)

        # Create mock file_store
        mock_file_store = MagicMock()
        mock_file_store.list_files = AsyncMock(return_value=[])
        mock_file_store.get_file_chunks = AsyncMock(return_value=[])

        watcher = BaseFileWatcher(
            watch_paths=str(temp_dir),
            scan_on_start=False,
            callback=callback,
            file_store=mock_file_store,
        )

        await watcher.start()
        await asyncio.sleep(0.1)
        await watcher.close()

        # No callback should be called for existing files
        assert len(callback_called) == 0

    @pytest.mark.asyncio
    async def test_scan_on_start_true_with_files(self, temp_files, temp_dir: Path):
        """Test that scan_on_start=True scans existing files."""
        callback_called = []

        async def callback(changes):
            callback_called.append(changes)

        # Create mock file_store
        mock_file_store = MagicMock()
        mock_file_store.list_files = AsyncMock(return_value=[])
        mock_file_store.get_file_chunks = AsyncMock(return_value=[])

        watcher = BaseFileWatcher(
            watch_paths=str(temp_dir),
            scan_on_start=True,
            callback=callback,
            file_store=mock_file_store,
        )

        await watcher.start()
        await asyncio.sleep(0.1)
        await watcher.close()

        # Callback should be called with existing files
        assert len(callback_called) >= 1

        # Check that files were detected as Change.added
        all_changes = set()
        for change_set in callback_called:
            all_changes.update(change_set)

        assert all(change == Change.added for change, _ in all_changes)

    @pytest.mark.asyncio
    async def test_scan_on_start_with_suffix_filter(self, temp_files, temp_dir: Path):
        """Test scan_on_start respects suffix filters."""
        callback_called = []

        async def callback(changes):
            callback_called.append(changes)

        mock_file_store = MagicMock()
        mock_file_store.list_files = AsyncMock(return_value=[])
        mock_file_store.get_file_chunks = AsyncMock(return_value=[])

        watcher = BaseFileWatcher(
            watch_paths=str(temp_dir),
            scan_on_start=True,
            suffix_filters=[".txt"],
            callback=callback,
            file_store=mock_file_store,
        )

        await watcher.start()
        await asyncio.sleep(0.1)
        await watcher.close()

        # Check only .txt files were scanned
        if callback_called:
            all_changes = set()
            for change_set in callback_called:
                all_changes.update(change_set)

            for _, path in all_changes:
                assert path.endswith(".txt"), f"Expected .txt file, got {path}"

    @pytest.mark.asyncio
    async def test_scan_on_start_recursive(self, temp_nested_dir: Path):
        """Test scan_on_start with recursive=True."""
        callback_called = []

        async def callback(changes):
            callback_called.append(changes)

        mock_file_store = MagicMock()
        mock_file_store.list_files = AsyncMock(return_value=[])
        mock_file_store.get_file_chunks = AsyncMock(return_value=[])

        watcher = BaseFileWatcher(
            watch_paths=str(temp_nested_dir),
            scan_on_start=True,
            recursive=True,
            suffix_filters=[".txt"],
            callback=callback,
            file_store=mock_file_store,
        )

        await watcher.start()
        await asyncio.sleep(0.1)
        await watcher.close()

        # Should find files in nested directories
        if callback_called:
            all_changes = set()
            for change_set in callback_called:
                all_changes.update(change_set)

            paths = [path for _, path in all_changes]
            # Should find nested_file.txt
            nested_found = any("nested_file.txt" in p for p in paths)
            assert nested_found, "Should find files in nested directories"

    @pytest.mark.asyncio
    async def test_scan_on_start_non_recursive(self, temp_nested_dir: Path):
        """Test scan_on_start with recursive=False."""
        callback_called = []

        async def callback(changes):
            callback_called.append(changes)

        mock_file_store = MagicMock()
        mock_file_store.list_files = AsyncMock(return_value=[])
        mock_file_store.get_file_chunks = AsyncMock(return_value=[])

        watcher = BaseFileWatcher(
            watch_paths=str(temp_nested_dir),
            scan_on_start=True,
            recursive=False,
            suffix_filters=[".txt"],
            callback=callback,
            file_store=mock_file_store,
        )

        await watcher.start()
        await asyncio.sleep(0.1)
        await watcher.close()

        # Should NOT find files in nested directories
        if callback_called:
            all_changes = set()
            for change_set in callback_called:
                all_changes.update(change_set)

            paths = [path for _, path in all_changes]
            nested_found = any("nested_file.txt" in p for p in paths)
            assert not nested_found, "Should not find files in nested directories"

    @pytest.mark.asyncio
    async def test_scan_on_start_nonexistent_path(self):
        """Test scan_on_start with non-existent path."""
        callback_called = []

        async def callback(changes):
            callback_called.append(changes)

        mock_file_store = MagicMock()
        mock_file_store.list_files = AsyncMock(return_value=[])
        mock_file_store.get_file_chunks = AsyncMock(return_value=[])

        watcher = BaseFileWatcher(
            watch_paths="/nonexistent/path",
            scan_on_start=True,
            callback=callback,
            file_store=mock_file_store,
        )

        await watcher.start()
        await asyncio.sleep(0.1)
        await watcher.close()

        # No files should be found
        assert len(callback_called) == 0


# ==================== Test Dynamic Path Management ====================


class TestDynamicPathManagement:
    """Tests for dynamic path add/remove."""

    @pytest.mark.asyncio
    async def test_add_path_when_stopped(self, temp_dir: Path):
        """Test adding path when watcher is stopped."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        new_dir = temp_dir / "new_dir"
        new_dir.mkdir()

        await watcher.add_path(str(new_dir))

        assert str(new_dir) in watcher.watch_paths

    @pytest.mark.asyncio
    async def test_add_path_when_running(self, temp_dir: Path):
        """Test adding path when watcher is running (triggers restart)."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        await watcher.start()
        assert watcher.is_running()

        new_dir = temp_dir / "new_dir"
        new_dir.mkdir()

        await watcher.add_path(str(new_dir))

        assert str(new_dir) in watcher.watch_paths
        assert watcher.is_running()

        await watcher.close()

    @pytest.mark.asyncio
    async def test_add_duplicate_path(self, temp_dir: Path):
        """Test adding duplicate path is ignored."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        original_count = len(watcher.watch_paths)
        await watcher.add_path(str(temp_dir))

        assert len(watcher.watch_paths) == original_count

    @pytest.mark.asyncio
    async def test_remove_path(self, temp_dir: Path):
        """Test removing path."""
        sub1 = temp_dir / "sub1"
        sub2 = temp_dir / "sub2"
        sub1.mkdir()
        sub2.mkdir()

        watcher = BaseFileWatcher(watch_paths=[str(sub1), str(sub2)])

        await watcher.remove_path(str(sub1))

        assert str(sub1) not in watcher.watch_paths
        assert str(sub2) in watcher.watch_paths

    @pytest.mark.asyncio
    async def test_remove_nonexistent_path(self, temp_dir: Path):
        """Test removing path that's not in watch list."""
        watcher = BaseFileWatcher(watch_paths=str(temp_dir))

        original_paths = watcher.watch_paths.copy()
        await watcher.remove_path("/some/other/path")

        assert watcher.watch_paths == original_paths


# ==================== Test Configuration Options ====================


class TestConfigurationOptions:
    """Tests for various configuration options."""

    @pytest.mark.asyncio
    async def test_debounce_setting(self, temp_dir: Path):
        """Test debounce configuration."""
        watcher = BaseFileWatcher(
            watch_paths=str(temp_dir),
            debounce=1000,
        )

        assert watcher.debounce == 1000

    @pytest.mark.asyncio
    async def test_chunk_settings(self, temp_dir: Path):
        """Test chunk configuration."""
        watcher = BaseFileWatcher(
            watch_paths=str(temp_dir),
            chunk_tokens=500,
            chunk_overlap=100,
        )

        assert watcher.chunk_tokens == 500
        assert watcher.chunk_overlap == 100

    @pytest.mark.asyncio
    async def test_recursive_setting(self, temp_dir: Path):
        """Test recursive configuration."""
        watcher = BaseFileWatcher(
            watch_paths=str(temp_dir),
            recursive=True,
        )

        assert watcher.recursive is True

    @pytest.mark.asyncio
    async def test_kwargs_preserved(self, temp_dir: Path):
        """Test that extra kwargs are preserved."""
        watcher = BaseFileWatcher(
            watch_paths=str(temp_dir),
            custom_arg1="value1",
            custom_arg2=123,
        )

        assert watcher.kwargs.get("custom_arg1") == "value1"
        assert watcher.kwargs.get("custom_arg2") == 123


# ==================== Test Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_watch_single_file(self, temp_files):
        """Test watching a single file instead of directory."""
        file_path = temp_files["txt_0"]

        mock_file_store = MagicMock()
        mock_file_store.list_files = AsyncMock(return_value=[])
        mock_file_store.get_file_chunks = AsyncMock(return_value=[])

        callback_called = []

        async def callback(changes):
            callback_called.append(changes)

        watcher = BaseFileWatcher(
            watch_paths=str(file_path),
            scan_on_start=True,
            callback=callback,
            file_store=mock_file_store,
        )

        await watcher.start()
        await asyncio.sleep(0.1)
        await watcher.close()

        # Single file should be detected
        if callback_called:
            all_changes = set()
            for change_set in callback_called:
                all_changes.update(change_set)
            assert len(all_changes) == 1

    @pytest.mark.asyncio
    async def test_empty_directory(self, temp_dir: Path):
        """Test watching empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        mock_file_store = MagicMock()
        mock_file_store.list_files = AsyncMock(return_value=[])
        mock_file_store.get_file_chunks = AsyncMock(return_value=[])

        callback_called = []

        async def callback(changes):
            callback_called.append(changes)

        watcher = BaseFileWatcher(
            watch_paths=str(empty_dir),
            scan_on_start=True,
            callback=callback,
            file_store=mock_file_store,
        )

        await watcher.start()
        await asyncio.sleep(0.1)
        await watcher.close()

        # No files should be detected
        assert len(callback_called) == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_path(self, temp_dir: Path):
        """Test paths with special characters."""
        special_dir = temp_dir / "test dir with spaces"
        special_dir.mkdir()

        file_path = special_dir / "file with spaces.txt"
        file_path.write_text("content")

        watcher = BaseFileWatcher(watch_paths=str(special_dir))

        assert watcher.watch_filter(Change.added, str(file_path)) is True

    @pytest.mark.asyncio
    async def test_unicode_in_path(self, temp_dir: Path):
        """Test paths with unicode characters."""
        unicode_dir = temp_dir / "测试目录"
        unicode_dir.mkdir()

        file_path = unicode_dir / "文件.txt"
        file_path.write_text("内容")

        watcher = BaseFileWatcher(
            watch_paths=str(unicode_dir),
            suffix_filters=[".txt"],
        )

        assert watcher.watch_filter(Change.added, str(file_path)) is True


# ==================== Main Entry Point ====================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
