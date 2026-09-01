"""Thin job-activity tracker feeding the idle gate (F4.2).

Every job type wraps its real execution edge in the same ``track()`` context
manager, so activity bookkeeping has exactly one implementation:
``active_count`` counts overlapping invocations of a job, ``last_start`` marks
the most recent begin, and ``last_end`` is recorded only when the count drops
back to zero. Concurrent invocations of one job therefore cannot flip it idle
while others still run, and no job type can forget the paired end call.

State lives in process-local ``app_context.metadata``, so a restart always
leaves every job looking idle.
"""

import threading
import time
from contextlib import asynccontextmanager
from typing import Any

JOB_ACTIVITY_KEY = "__job_activity"


class JobActivityTracker:
    """Per-job activity state shared by all job types through ``metadata``."""

    def __init__(self, metadata: dict[str, Any]):
        self._metadata = metadata
        # Plain threading lock: background jobs may run on a thread-pool loop,
        # and the guarded sections never await.
        self._lock = threading.Lock()

    def _entries(self) -> dict[str, dict]:
        entries = self._metadata.get(JOB_ACTIVITY_KEY)
        if not isinstance(entries, dict):
            entries = {}
            self._metadata[JOB_ACTIVITY_KEY] = entries
        return entries

    @asynccontextmanager
    async def track(self, name: str):
        """Mark one real execution edge of job ``name``; begin/end always pair."""
        with self._lock:
            entry = self._entries().setdefault(name, {"active_count": 0, "last_start": None, "last_end": None})
            entry["active_count"] = int(entry.get("active_count") or 0) + 1
            entry["last_start"] = time.monotonic()
        try:
            yield
        finally:
            with self._lock:
                entry = self._entries().get(name)
                if isinstance(entry, dict):
                    count = max(int(entry.get("active_count") or 0) - 1, 0)
                    entry["active_count"] = count
                    if count == 0:
                        entry["last_end"] = time.monotonic()

    def snapshot(self) -> dict[str, dict]:
        """Copy of the current per-job activity state."""
        with self._lock:
            return {name: dict(entry) for name, entry in self._entries().items()}
