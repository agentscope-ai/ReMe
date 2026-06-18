"""Light tests for reme4 file I/O helpers."""

import asyncio
from pathlib import Path

from reme4.steps.file_io._file_io import read_file_lines_safe
from reme4.steps.file_io._path import resolve_path


def test_resolve_path_accepts_absolute_and_rejects_parent_escape(tmp_path: Path):
    """Absolute paths remain compatible; relative parent escapes are rejected."""
    vault = tmp_path / "vault"
    vault.mkdir()

    absolute = str(tmp_path / "outside.md")
    target, err = resolve_path(vault, absolute)
    assert err is None
    assert target == Path(absolute).resolve()
    assert resolve_path(vault, "../outside.md")[1] == "path component cannot be '.' or '..': '..'"


def test_resolve_path_allows_empty_when_requested(tmp_path: Path):
    """Empty paths can resolve to the vault root for list-like operations."""
    vault = tmp_path / "vault"
    vault.mkdir()

    target, err = resolve_path(vault, "", allow_empty=True)

    assert err is None
    assert target == vault.resolve()


def test_read_file_lines_safe_counts_full_file_but_limits_collected_text(tmp_path: Path):
    """Large-file reader counts all lines while bounding returned text."""
    path = tmp_path / "large.md"
    path.write_text("\n".join(f"line {i}" for i in range(1, 101)), encoding="utf-8")

    text, total, encoding = asyncio.run(read_file_lines_safe(path, 10, None, max_collect_bytes=25))

    assert total == 100
    assert encoding == "utf-8"
    assert text.startswith("line 10\nline 11")
    assert "line 100" not in text
