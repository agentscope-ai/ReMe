"""``cron_step`` — periodically dispatch downstream step(s) or job(s) from a BackgroundJob.

Long-running step. Sleeps until the next fire time, dispatches its
configured downstream step(s) or job(s), loops. Exits when
``context.stop_event`` is set, so the surrounding ``BackgroundJob``
supervisor can shut it down cleanly.

Two dispatch modes — pick one (or both, executed in order: jobs first,
then steps):

* ``dispatch_job`` / ``dispatch_jobs`` — invoke a **registered job**
  by name through ``self.run_job(name, **extra)``. Goes through the
  job's parameter validation and context construction, identical to a
  CLI / MCP invocation. Prefer this when the periodic task already
  has a job wrapping it (e.g. ``auto_dream``).
* ``dispatch_step`` / ``dispatch_steps`` — instantiate a **registered
  step** by name and invoke it directly with the inherited
  ``context.data`` (sans ``stop_event``). Use for steps that don't
  have a job wrapper.

Three schedule modes (mutually exclusive — exactly one must be set):

* ``cron: "M H DoM Mon DoW"`` — standard 5-field cron expression in
  ``app_config.timezone``. Most flexible; use for non-daily cadence
  (``"0 */6 * * *"`` = every 6 hours, ``"0 3 * * 1-5"`` = 3am on
  weekdays, ``"*/15 * * * *"`` = every 15 minutes).
* ``daily_at: "HH:MM"`` — fire once per day at this wall-clock time, in
  ``app_config.timezone``. Convenience shorthand for ``"M H * * *"``.
* ``interval_seconds: int`` — fire every N seconds since launch. Useful
  for tests and for time-zone-independent sub-minute cadence.

Per-dispatch exceptions are caught and logged — a failed downstream
job or step never kills the cron loop. ``run_on_start: true`` triggers
one immediate dispatch on launch (default ``false``, to avoid a burst
when ``Application.start`` brings several cron jobs up at once).

Cron expressions are evaluated via ``croniter`` and validated eagerly at
construction time, so a typo fails at app start rather than at 3 a.m.
"""

import asyncio
import datetime
import zoneinfo

from croniter import croniter

from ..base_step import BaseStep
from ...components import R, BaseComponent
from ...enumeration import ComponentEnum


@R.register("cron_step")
class CronStep(BaseStep):
    """Dispatch downstream step(s) and/or job(s) on a schedule until ``stop_event`` fires."""

    def __init__(
        self,
        dispatch_step: str = "",
        dispatch_steps: list[str] | None = None,
        dispatch_job: str = "",
        dispatch_jobs: list[str] | None = None,
        cron: str = "",
        daily_at: str = "",
        interval_seconds: int = 0,
        run_on_start: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dispatch_steps: list[str] = dispatch_steps or ([dispatch_step] if dispatch_step else [])
        self.dispatch_jobs: list[str] = dispatch_jobs or ([dispatch_job] if dispatch_job else [])
        self.cron: str = cron
        self.daily_at: str = daily_at
        self.interval_seconds: int = interval_seconds
        self.run_on_start: bool = run_on_start

        if not self.dispatch_steps and not self.dispatch_jobs:
            raise ValueError(
                "cron_step requires at least one of 'dispatch_step'/'dispatch_steps' "
                "or 'dispatch_job'/'dispatch_jobs'",
            )

        schedules_set = sum(bool(x) for x in (self.cron, self.daily_at, self.interval_seconds))
        if schedules_set != 1:
            raise ValueError(
                "cron_step requires exactly one of "
                "'cron' (5-field expression), 'daily_at' (HH:MM), "
                "or 'interval_seconds' (int)",
            )

        if self.daily_at:
            # Validate HH:MM format eagerly so misconfig fails at start, not at 3am.
            h, m = self._parse_hh_mm(self.daily_at)
            self._fire_hour, self._fire_minute = h, m
        elif self.cron:
            # Fail at start, not at the next scheduled tick.
            if not croniter.is_valid(self.cron):
                raise ValueError(f"cron expression invalid, got {self.cron!r}")

    @staticmethod
    def _parse_hh_mm(value: str) -> tuple[int, int]:
        try:
            h_str, m_str = value.split(":", 1)
            h, m = int(h_str), int(m_str)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"daily_at must be 'HH:MM', got {value!r}") from exc
        if not (0 <= h < 24 and 0 <= m < 60):
            raise ValueError(f"daily_at out of range, got {value!r}")
        return h, m

    def _tz(self) -> datetime.tzinfo | None:
        if not self.app_context:
            return None
        tz_name = self.app_context.app_config.timezone
        if not tz_name:
            return None
        try:
            return zoneinfo.ZoneInfo(tz_name)
        except zoneinfo.ZoneInfoNotFoundError:
            self.logger.warning(f"[{self.name}] unknown timezone {tz_name!r}; using local time")
            return None

    def _next_fire_delay(self) -> float:
        """Seconds from now until the next fire — never negative, never zero."""
        if self.interval_seconds:
            return float(self.interval_seconds)
        tz = self._tz()
        now = datetime.datetime.now(tz)
        if self.cron:
            # croniter requires an explicit base time; tz comes through on `now`.
            nxt = croniter(self.cron, now).get_next(datetime.datetime)
            return (nxt - now).total_seconds()
        # daily_at
        target = now.replace(hour=self._fire_hour, minute=self._fire_minute, second=0, microsecond=0)
        if target <= now:
            target = target + datetime.timedelta(days=1)
        return (target - now).total_seconds()

    async def _wait_or_stop(self, delay: float, stop_event: asyncio.Event) -> bool:
        """Sleep up to ``delay`` seconds. Returns True if stop_event fired."""
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            return False
        return True

    async def _fire(self, dispatch_classes: list[type[BaseComponent]]) -> None:
        """Dispatch each downstream job (via run_job) and step (class-level), in that order;
        swallow exceptions per-dispatch so a single failure doesn't break the loop."""
        assert self.context is not None
        # Inherit cron_step's own context data so downstream callees can read
        # shared bits, except stop_event (each runs its own lifecycle).
        extra = {k: v for k, v in self.context.data.items() if k != "stop_event"}

        # Jobs first — they're the higher-level invocation path (matches CLI / MCP).
        for name in self.dispatch_jobs:
            try:
                await self.run_job(name, **extra)
                self.logger.info(f"[{self.name}] dispatched job {name!r}")
            except Exception as exc:
                self.logger.exception(f"[{self.name}] dispatch job {name!r} raised: {exc}")

        for cls in dispatch_classes:
            try:
                s = cls(app_context=self.app_context)
                await s(**extra)
                self.logger.info(f"[{self.name}] dispatched step {cls.__name__}")
            except Exception as exc:
                self.logger.exception(f"[{self.name}] dispatch step {cls.__name__} raised: {exc}")

    async def execute(self):
        assert self.context is not None
        if self.context.stop_event is None:
            raise RuntimeError("cron_step requires 'stop_event' on context")
        stop_event: asyncio.Event = self.context.stop_event

        dispatch_classes: list[type[BaseComponent]] = []
        for name in self.dispatch_steps:
            cls = R.get(ComponentEnum.STEP, name)
            if cls is None:
                raise RuntimeError(f"Unregistered step '{name}'")
            dispatch_classes.append(cls)

        if self.cron:
            mode = f"cron={self.cron!r}"
        elif self.daily_at:
            mode = f"daily_at={self.daily_at}"
        else:
            mode = f"interval_seconds={self.interval_seconds}"
        self.logger.info(
            f"[{self.name}] cron loop start "
            f"dispatch_jobs={self.dispatch_jobs} dispatch_steps={self.dispatch_steps} "
            f"{mode} run_on_start={self.run_on_start}",
        )

        if self.run_on_start and not stop_event.is_set():
            await self._fire(dispatch_classes)

        while not stop_event.is_set():
            delay = self._next_fire_delay()
            self.logger.info(f"[{self.name}] next fire in {delay:.0f}s")
            stopped = await self._wait_or_stop(delay, stop_event)
            if stopped:
                break
            await self._fire(dispatch_classes)

        self.context.response.success = True
        self.context.response.answer = f"cron_step jobs={self.dispatch_jobs!r} steps={self.dispatch_steps!r} stopped"
        return self.context.response
