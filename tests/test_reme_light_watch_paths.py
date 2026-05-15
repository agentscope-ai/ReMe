"""
Tests for the default watch path construction in ``ReMeLight``.

These tests verify that the built-in ``MEMORY.md`` / ``memory.md`` / memory
directory watch list does not produce duplicate entries on case-insensitive
filesystems (Windows NTFS, macOS HFS+), where the two ``.md`` paths refer to
the same physical file. See agentscope-ai/ReMe#228.
"""

# pylint: disable=redefined-outer-name,protected-access,unused-argument

import os.path
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from reme.reme_light import ReMeLight


@pytest.fixture
def temp_working_dir():
    """Provide an isolated working directory per test."""
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def _captured_watch_paths(working_dir: str, *, default_file_watcher_config=None):
    """Construct a ``ReMeLight`` and return the watch_paths the parent
    ``Application.__init__`` would have received.

    The parent ``Application.__init__`` is patched to a no-op that simply
    captures its keyword arguments, so this test exercises the watch-path
    construction logic in ``ReMeLight.__init__`` without spinning up the full
    application stack.
    """
    captured: dict = {}

    def _capture(self, *_args, **kwargs):
        captured.update(kwargs)

    with patch("reme.reme_light.Application.__init__", _capture):
        ReMeLight(
            working_dir=working_dir,
            default_file_watcher_config=default_file_watcher_config,
        )

    watcher_config = captured.get("default_file_watcher_config") or {}
    return list(watcher_config.get("watch_paths", []))


class TestDefaultWatchPathDeduplication:
    """Regression tests for issue #228."""

    def test_case_insensitive_filesystem_dedupes_memory_md_variants(self, temp_working_dir):
        """On a case-insensitive filesystem, ``MEMORY.md`` and ``memory.md``
        resolve to the same file and must not both appear in the default
        watch list. We simulate this by patching ``os.path.normcase`` to the
        Windows-style lower-casing behavior.
        """
        with patch.object(os.path, "normcase", lambda p: p.lower()):
            paths = _captured_watch_paths(temp_working_dir)

        # Expect the memory directory plus exactly one of the two markdown
        # spellings. Both markdown names normcase to the same key, so the
        # second occurrence is dropped.
        memory_dir = str(Path(temp_working_dir).absolute() / "memory")
        markdown_paths = [p for p in paths if p.lower().endswith("memory.md")]

        assert memory_dir in paths
        assert len(markdown_paths) == 1, (
            "Expected MEMORY.md/memory.md to be deduplicated on a "
            f"case-insensitive filesystem, got {markdown_paths!r}"
        )
        assert len(paths) == 2

    def test_case_sensitive_filesystem_keeps_both_memory_md_variants(self, temp_working_dir):
        """On a case-sensitive filesystem the two markdown spellings can refer
        to distinct files, so both must be preserved.
        """
        # POSIX-style normcase is the identity function.
        with patch.object(os.path, "normcase", lambda p: p):
            paths = _captured_watch_paths(temp_working_dir)

        upper = str(Path(temp_working_dir).absolute() / "MEMORY.md")
        lower = str(Path(temp_working_dir).absolute() / "memory.md")
        memory_dir = str(Path(temp_working_dir).absolute() / "memory")

        assert upper in paths
        assert lower in paths
        assert memory_dir in paths
        assert len(paths) == 3

    def test_user_provided_watch_paths_are_passed_through(self, temp_working_dir):
        """When the caller supplies its own ``watch_paths``, the defaults are
        not used at all, regardless of filesystem behavior.
        """
        custom = [str(Path(temp_working_dir) / "notes.md")]
        with patch.object(os.path, "normcase", lambda p: p.lower()):
            paths = _captured_watch_paths(
                temp_working_dir,
                default_file_watcher_config={"watch_paths": custom},
            )

        assert paths == custom

    def test_dedup_preserves_original_path_casing(self, temp_working_dir):
        """The deduplication step must not mutate the surviving entry into
        its lower-cased form; the path that ``Application`` receives should
        still be the original ``MEMORY.md`` (or ``memory.md``) string.
        """
        with patch.object(os.path, "normcase", lambda p: p.lower()):
            paths = _captured_watch_paths(temp_working_dir)

        markdown_paths = [p for p in paths if p.lower().endswith("memory.md")]
        assert len(markdown_paths) == 1
        # The first candidate in the source list is ``MEMORY.md``, so that
        # spelling is the one that should survive.
        assert markdown_paths[0].endswith("MEMORY.md")
