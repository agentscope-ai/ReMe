"""Shared file-version checks used by file readers and index updates."""

import os
from pathlib import Path


class FileChangedError(RuntimeError):
    """Raised when the source file no longer matches a recorded version."""


def check_file_version(path: Path, before: os.stat_result, after: os.stat_result) -> None:
    """Raise when the source file's identity or content metadata changed."""
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise FileChangedError(f"File changed: {path}")
