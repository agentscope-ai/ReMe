"""Cron-scheduled background job that runs its configured steps."""

import datetime
from zoneinfo import ZoneInfo

from .background_job import BackgroundJob
from ...core.registry import R
from ...core.context import RuntimeContext
from ...schema import ComponentConfig, Response
from ...utils.cron_utils import next_cron_time
from ...utils.timezone_utils import system_timezone


@R.register("cron")
class CronJob(BackgroundJob):
    """Run optional idempotent init steps, then schedule fresh steps on a cron expression."""

    def __init__(self, cron: str, init_steps: list[ComponentConfig | dict] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.cron_expr = cron
        self.init_step_configs = init_steps or []
        self.init_step_specs = []

    async def _start(self) -> None:
        from croniter import croniter

        if not croniter.is_valid(self.cron_expr):
            raise ValueError(f"Invalid cron expression: {self.cron_expr}")
        if self.app_context is None:
            raise ValueError(f"app_context must be provided for job '{self.name}'")
        self.init_step_specs = [self._resolve_step(raw) for raw in self.init_step_configs]
        await super()._start()

    async def _close(self) -> None:
        await super()._close()
        self.init_step_specs.clear()

    def _next_fire_delay(self) -> float:
        tz_name = self.app_context.app_config.timezone if self.app_context is not None else None
        now = datetime.datetime.now(ZoneInfo(tz_name) if tz_name else system_timezone())
        next_fire = next_cron_time(self.cron_expr, now)
        return max(0.0, next_fire.timestamp() - now.timestamp())

    async def _execute_steps(self) -> Response:
        context = RuntimeContext(stop_event=self._stop_event, **self.kwargs)
        with self._execution(context):
            return await self._run_steps(context)

    async def _initialize(self) -> None:
        if not self.init_step_specs or self._stop_event.is_set():
            return
        context = RuntimeContext(stop_event=self._stop_event, **self.kwargs)
        with self._execution(context):
            for step_cls, params in self.init_step_specs:
                if self._stop_event.is_set():
                    break
                await step_cls(**dict(params))(context)
                if not context.response.success:
                    raise RuntimeError(f"Cron job '{self.name}' initialization failed: {context.response.answer}")

    async def _run_once(self) -> None:
        # Initialization and individual triggers are recorded; waiting is not.
        await self()

    async def __call__(self, **kwargs) -> Response:
        self.check_enabled()
        assert self._stop_event is not None
        await self._initialize()
        while not self._stop_event.is_set():
            await self._wait_or_stop(self._next_fire_delay())
            if self._stop_event.is_set():
                break
            try:
                result = await self._execute_steps()
                if not result.success:
                    self.logger.error(f"Cron job '{self.name}' failed: {result.answer}")
            except Exception as exc:
                self.logger.exception(f"Cron job '{self.name}' failed: {exc}")
        response = Response()
        response.success = True
        return response
