"""Shared per-path locks and file-version checks."""

import asyncio
import os
from pathlib import Path

_PATH_LOCKS_MAX = 1024
_PATH_LOCKS: dict[str, asyncio.Lock] = {}
_PATH_LOCKS_REGISTRY = asyncio.Lock()


async def get_path_lock(target: Path) -> asyncio.Lock:
    """Return the shared in-process write lock for a workspace path."""
    key = str(target)
    async with _PATH_LOCKS_REGISTRY:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            if len(_PATH_LOCKS) >= _PATH_LOCKS_MAX:
                to_remove = [k for k, v in _PATH_LOCKS.items() if not v.locked()]
                for k in to_remove[: len(_PATH_LOCKS) // 2]:
                    del _PATH_LOCKS[k]
            lock = asyncio.Lock()
            _PATH_LOCKS[key] = lock
    return lock


class FileChangedDuringChunkingError(RuntimeError):
    """Raised when a source file changes while it is being parsed."""


def check_file_version(path: Path, before: os.stat_result, after: os.stat_result) -> None:
    """Raise when the source file's identity or content metadata changed."""
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise FileChangedDuringChunkingError(f"File changed while chunking: {path}")
