"""Long-running background job with optional supervisor."""

import asyncio
import random

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
    """

    def __init__(
        self,
        supervisor: bool = True,
        backoff_base: float = 1.0,
        backoff_cap: float = 60.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supervisor: bool = supervisor
        self.backoff_base: float = backoff_base
        self.backoff_cap: float = backoff_cap
        self._stop_event: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _start(self) -> None:
        await super()._start()
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_with_supervisor())

    async def _close(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await self._task
            except Exception:
                self.logger.exception(f"Background task '{self.name}' raised during close")
            self._task = None
        await super()._close()

    async def _run_with_supervisor(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            try:
                await self()
                return
            except Exception as e:
                if not self.supervisor:
                    raise
                delay = min(self.backoff_base * (2**attempt), self.backoff_cap) * (0.5 + random.random())
                self.logger.exception(f"job body crashed, restart in {delay:.2f}s error={e}")
                attempt += 1
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass

    async def __call__(self, **kwargs) -> Response:
        """Default body: run step_components in order; errors propagate to supervisor."""
        context = RuntimeContext(stop_event=self._stop_event, **self.kwargs)
        for step in self.step_components:
            await step(context)
        return context.response
