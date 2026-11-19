"""Shared in-process write locks for workspace files and runtime bookkeeping."""

import asyncio
from pathlib import Path

_PATH_LOCKS_MAX = 1024
_PATH_LOCKS: dict[str, asyncio.Lock] = {}
_PATH_LOCKS_REGISTRY = asyncio.Lock()


async def get_path_lock(target: Path) -> asyncio.Lock:
    """Return the asyncio.Lock for ``target``; created lazily on first request."""
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
