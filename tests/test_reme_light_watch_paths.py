"""
Tests for the default watch path construction in ``ReMeLight``.

Verifies that the built-in watch list picks a single ``MEMORY.md`` /
``memory.md`` spelling so the file is not indexed twice on case-insensitive
filesystems (Windows NTFS, macOS APFS/HFS+). See agentscope-ai/ReMe#228.
"""

# pylint: disable=redefined-outer-name,protected-access,missing-function-docstring,missing-class-docstring
# pylint: disable=no-name-in-module

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from reme.reme_light import ReMeLight


@pytest.fixture
def temp_working_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def _captured_watch_paths(working_dir: str, *, default_file_watcher_config=None):
    """Capture the ``watch_paths`` that ``ReMeLight`` would forward to its
    parent ``Application.__init__``, without spinning up the full app stack."""
    captured: dict = {}

    def _capture(*_args, **kwargs):
        captured.update(kwargs)

    with patch("reme.reme_light.Application.__init__", _capture):
        ReMeLight(
            working_dir=working_dir,
            default_file_watcher_config=default_file_watcher_config,
        )

    return list((captured.get("default_file_watcher_config") or {}).get("watch_paths", []))


class TestDefaultWatchPaths:
    def test_defaults_to_uppercase_memory_md_when_neither_exists(self, temp_working_dir):
        paths = _captured_watch_paths(temp_working_dir)
        memory_dir = str(Path(temp_working_dir).absolute() / "memory")
        assert paths == [str(Path(temp_working_dir).absolute() / "MEMORY.md"), memory_dir]

    def test_picks_lowercase_memory_md_when_only_it_exists(self, temp_working_dir):
        (Path(temp_working_dir) / "memory.md").write_text("")
        if (Path(temp_working_dir) / "MEMORY.md").exists():
            pytest.skip("case-insensitive filesystem treats both spellings as one file")
        paths = _captured_watch_paths(temp_working_dir)
        assert paths[0] == str(Path(temp_working_dir).absolute() / "memory.md")

    def test_prefers_uppercase_memory_md_when_it_exists(self, temp_working_dir):
        (Path(temp_working_dir) / "MEMORY.md").write_text("")
        paths = _captured_watch_paths(temp_working_dir)
        assert paths[0] == str(Path(temp_working_dir).absolute() / "MEMORY.md")

    def test_user_provided_watch_paths_pass_through(self, temp_working_dir):
        custom = [str(Path(temp_working_dir) / "notes.md")]
        paths = _captured_watch_paths(
            temp_working_dir,
            default_file_watcher_config={"watch_paths": custom},
        )
        assert paths == custom
