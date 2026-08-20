"""Idle gate: wait until trunk jobs are quiet before running low-priority work (F4.2)."""

import asyncio
import fnmatch
import time

from ..base_step import BaseStep
from ...components import R

DEFAULT_BUSY_JOB_PATTERNS = ["auto_memory*", "auto_resource*", "dream_cron", "optimize_index_cron"]


@R.register("wait_for_idle_step")
class WaitForIdleStep(BaseStep):
    """Hold until all busy-trunk jobs are idle; give up the round after ``max_wait``.

    Giving up is not a failure: ``response.success`` stays True and the
    short-circuit flag ``context[skip_key] = {"reason": "busy"}`` lets the
    downstream steps of this job pass through (F4.3).
    """

    def __init__(
        self,
        busy_job_patterns: list[str] | None = None,
        quiet_window: float = 120,
        poll_interval: float = 10,
        max_wait: float = 600,
        skip_key: str = "proactive_skip",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.busy_job_patterns = list(busy_job_patterns or DEFAULT_BUSY_JOB_PATTERNS)
        self.quiet_window = float(quiet_window)
        self.poll_interval = max(float(poll_interval), 0.1)
        self.max_wait = float(max_wait)
        self.skip_key = skip_key

    async def execute(self):
        assert self.context is not None
        metadata = getattr(self.app_context, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        stop_event = getattr(self.context, "stop_event", None)
        deadline = time.monotonic() + self.max_wait
        self.logger.info(
            f"[{self.name}] start patterns={self.busy_job_patterns} quiet_window={self.quiet_window}s "
            f"max_wait={self.max_wait}s skip_key={self.skip_key}",
        )
        while True:
            busy = self._busy_jobs(metadata)
            if not busy:
                self.logger.info(f"[{self.name}] trunk idle; proceeding")
                self.context.response.success = True
                self.context.response.answer = "Trunk idle; proceeding"
                return self.context.response
            if stop_event is not None and stop_event.is_set():
                self.logger.info(f"[{self.name}] stop requested while waiting; giving up this round")
                return self._give_up(busy, "stop_requested")
            if time.monotonic() >= deadline:
                self.logger.info(f"[{self.name}] still busy after max_wait={self.max_wait}s: {busy}")
                return self._give_up(busy, "busy")
            self.logger.info(f"[{self.name}] trunk busy {busy}; polling again in {self.poll_interval}s")
            await self._sleep_or_stop(stop_event, self.poll_interval)

    def _give_up(self, busy: list[str], reason: str):
        assert self.context is not None
        self.context[self.skip_key] = {"reason": reason, "busy_jobs": busy}
        self.context.response.success = True
        self.context.response.answer = f"Skipped: trunk busy ({', '.join(busy)}); giving up this round"
        return self.context.response

    async def _sleep_or_stop(self, stop_event, delay: float) -> None:
        if isinstance(stop_event, asyncio.Event):
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            return
        await asyncio.sleep(delay)

    def _busy_jobs(self, metadata: dict) -> list[str]:
        """Return names of matching jobs that are running or inside the quiet window."""
        last_run = metadata.get("__job_last_run") or {}
        now = time.monotonic()
        busy: list[str] = []
        for name, info in last_run.items():
            if not isinstance(info, dict):
                continue
            if not any(fnmatch.fnmatch(name, pattern) for pattern in self.busy_job_patterns):
                continue
            last_end = info.get("last_end")
            recently_active = isinstance(last_end, (int, float)) and now - float(last_end) <= self.quiet_window
            if info.get("running") or recently_active:
                busy.append(name)
        return sorted(busy)
