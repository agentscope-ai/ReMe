"""Shared in-process write locks for workspace files and runtime bookkeeping."""

import asyncio
from pathlib import Path
from weakref import WeakValueDictionary

_PATH_LOCKS: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_PATH_LOCKS_REGISTRY = asyncio.Lock()


async def get_path_lock(target: Path) -> asyncio.Lock:
    """Return the asyncio.Lock for ``target``; created lazily on first request."""
    key = str(target)
    async with _PATH_LOCKS_REGISTRY:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _PATH_LOCKS[key] = lock
    return lock
