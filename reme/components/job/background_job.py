"""Long-running background job with optional supervisor."""

import asyncio
import contextlib
import random
import threading
import time

from .base_job import BaseJob
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...schema import Response


@R.register("background")
class BackgroundJob(BaseJob):
    """Long-running job started by Application._start; runs __call__ until close.

    Subclasses override __call__ (or use the default step-based body). If
    supervisor=True (default) and __call__ raises, it is restarted with
    exponential backoff (backoff_base * 2**attempt, capped at backoff_cap)
    plus ±50% jitter. __call__ must NOT swallow exceptions, otherwise the
    supervisor cannot trigger a restart.

    On close, the stop_event is set and the task is given up to
    ``close_timeout`` seconds to exit gracefully; after that it is cancelled.
    If a single run survives at least ``attempt_reset_after`` seconds before
    crashing, the backoff attempt counter resets — so a long-stable job that
    eventually crashes restarts quickly rather than at the capped delay.
    """

    def __init__(
        self,
        supervisor: bool = True,
        backoff_base: float = 1.0,
        backoff_cap: float = 60.0,
        close_timeout: float = 5.0,
        attempt_reset_after: float = 60.0,
        use_thread_pool: bool = False,
        **kwargs,
    ):
        # Background jobs are long-running loops, not request-shaped callables —
        # forced off so they never get registered as service tools.
        kwargs.pop("enable_serve", None)
        super().__init__(enable_serve=False, **kwargs)
        self.supervisor: bool = supervisor
        self.backoff_base: float = backoff_base
        self.backoff_cap: float = backoff_cap
        self.close_timeout: float = close_timeout
        self.attempt_reset_after: float = attempt_reset_after
        self.use_thread_pool: bool = use_thread_pool
        self._stop_event: asyncio.Event | threading.Event | None = None
        self._task: asyncio.Task | None = None

    async def _start(self) -> None:
        await super()._start()
        if self.use_thread_pool and self.app_context.thread_pool:
            self._stop_event = threading.Event()
            self._task = asyncio.get_event_loop().run_in_executor(
                self.app_context.thread_pool,
                lambda: asyncio.run(self._run_with_supervisor()),
            )
        else:
            self._stop_event = asyncio.Event()
            self._task = asyncio.create_task(self._run_with_supervisor())

    async def _close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        await self._shutdown_task()
        await super()._close()

    async def _shutdown_task(self) -> None:
        """Wait close_timeout for graceful exit, then force-cancel."""
        if self._task is None:
            return
        try:
            # shield prevents wait_for's cancellation from propagating to the task itself,
            # so a timeout here truly times out instead of cancelling silently.
            await asyncio.wait_for(asyncio.shield(self._task), timeout=self.close_timeout)
        except asyncio.TimeoutError:
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task
        except Exception:
            self.logger.exception(f"Background task '{self.name}' raised during close")
        self._task = None

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with ±50% jitter, capped at backoff_cap."""
        capped = min(self.backoff_base * (2**attempt), self.backoff_cap)
        return min(capped * (0.5 + random.random()), self.backoff_cap)

    async def _wait_or_stop(self, delay: float) -> None:
        """Sleep up to delay, returning immediately when stop_event is set."""
        assert self._stop_event is not None
        if isinstance(self._stop_event, threading.Event):
            self._stop_event.wait(timeout=delay)
        else:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def _run_with_supervisor(self) -> None:
        assert self._stop_event is not None
        attempt = 0
        while not self._stop_event.is_set():
            started_at = time.monotonic()
            try:
                await self()
                return
            except Exception as e:
                if not self.supervisor:
                    raise
                # A long-stable run that just crashed restarts fresh rather than at the capped delay.
                if time.monotonic() - started_at >= self.attempt_reset_after:
                    attempt = 0
                delay = self._backoff_delay(attempt)
                self.logger.exception(f"job body crashed, restart in {delay:.2f}s error={e}")
                attempt += 1
                await self._wait_or_stop(delay)

    async def __call__(self, **kwargs) -> Response:
        """Default body: run steps in order; errors propagate to supervisor."""
        merged = {**self.kwargs, **kwargs}
        context = RuntimeContext(stop_event=self._stop_event, **merged)
        for step in self._build_steps():
            await step(context)
        return context.response
