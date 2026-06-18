"""Cron-scheduled background job that runs its configured steps."""

import datetime
from zoneinfo import ZoneInfo

from .background_job import BackgroundJob
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...schema import Response


@R.register("cron")
class CronJob(BackgroundJob):
    """Run this job's own steps on a cron expression."""

    def __init__(self, cron: str, **kwargs):
        super().__init__(**kwargs)
        self.cron_expr = cron

    async def _start(self) -> None:
        from croniter import croniter

        if not croniter.is_valid(self.cron_expr):
            raise ValueError(f"Invalid cron expression: {self.cron_expr}")
        await super()._start()

    def _next_fire_delay(self) -> float:
        tz_name = None
        if self.app_context is not None:
            tz_name = self.app_context.app_config.timezone
        now = datetime.datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.datetime.now()
        from croniter import croniter

        nxt = croniter(self.cron_expr, now).get_next(datetime.datetime)
        return max(0.0, (nxt - now).total_seconds())

    async def _execute_steps(self) -> Response:
        context = RuntimeContext(**self.kwargs)
        for step in self._build_steps():
            await step(context)
        return context.response

    async def __call__(self, **kwargs) -> Response:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            await self._wait_or_stop(self._next_fire_delay())
            if self._stop_event.is_set():
                break
            try:
                await self._execute_steps()
            except Exception as exc:
                self.logger.exception(f"Cron job '{self.name}' failed: {exc}")
        response = Response()
        response.success = True
        return response
