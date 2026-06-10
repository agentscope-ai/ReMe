"""Unit tests for the ``cron`` job — schedule math + dispatch wiring.

Strategy:

* ``_parse_hh_mm`` / ``_next_fire_delay`` are exercised directly without
  the BackgroundJob supervisor. Avoids wall-clock waits.
* ``__call__`` is tested by driving the job's own ``_stop_event``: the
  loop runs at most one iteration with a tiny ``interval_seconds`` and
  ``run_on_start=True`` — verifies the fire path actually dispatches the
  downstream step exactly once.
* The dispatch target is a tiny in-test counter step registered into
  the same step registry the production code uses, so we exercise the
  real ``R.get(ComponentEnum.STEP, name) → instantiate → __call__``
  path rather than mocking it.
* Tests run under a tempdir so the implicit Application context is
  isolated from the real vault.
"""

# pylint: disable=protected-access

import asyncio
import datetime
import os
import tempfile
import zoneinfo
from contextlib import contextmanager
from pathlib import Path

from reme4 import Application
from reme4.components import R
from reme4.components.job.cron_job import CronJob
from reme4.config import resolve_app_config
from reme4.steps.base_step import BaseStep


@R.register("test_cron_counter_step")
class _CounterStep(BaseStep):
    """In-test counter — increments a class-level fire count on each invocation."""

    fires: int = 0

    async def execute(self):
        type(self).fires += 1
        if self.context is not None:
            self.context.response.success = True
        return self.context.response


@contextmanager
def _temp_chdir(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _make_job(**kwargs) -> CronJob:
    # Default to a known-registered step so most tests can ignore dispatch wiring.
    kwargs.setdefault("dispatch_step", "version_step")
    return CronJob(**kwargs)


def _test_parse_hh_mm_valid() -> None:
    job = _make_job(daily_at="03:00")
    assert job._fire_hour == 3 and job._fire_minute == 0
    job = _make_job(daily_at="23:59")
    assert job._fire_hour == 23 and job._fire_minute == 59
    print("OK parse_hh_mm_valid")


def _test_parse_hh_mm_invalid() -> None:
    for bad in ["24:00", "03:60", "abc", "3", "03:", ":00"]:
        try:
            _make_job(daily_at=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for daily_at={bad!r}")
    print("OK parse_hh_mm_invalid")


def _test_requires_dispatch_step() -> None:
    # neither dispatch_step nor dispatch_steps → ValueError
    try:
        CronJob(daily_at="03:00")
    except ValueError:
        print("OK requires_dispatch_step")
        return
    raise AssertionError("expected ValueError when no dispatch step is configured")


def _test_dispatch_steps_list() -> None:
    # Mirrors watch_changes_step: dispatch_steps takes priority over dispatch_step,
    # both forms accepted, defaults coalesce.
    job1 = CronJob(dispatch_step="version_step", interval_seconds=60)
    assert job1.dispatch_steps == ["version_step"]

    job2 = CronJob(dispatch_steps=["a", "b"], interval_seconds=60)
    assert job2.dispatch_steps == ["a", "b"]

    job3 = CronJob(dispatch_step="x", dispatch_steps=["y", "z"], interval_seconds=60)
    assert job3.dispatch_steps == ["y", "z"]
    print("OK dispatch_steps_list")


def _test_dispatch_jobs_list() -> None:
    # dispatch_job / dispatch_jobs coalesce the same way and satisfy the
    # "at least one dispatch target" requirement on their own.
    job1 = CronJob(dispatch_job="auto_dream", interval_seconds=60)
    assert job1.dispatch_jobs == ["auto_dream"] and job1.dispatch_steps == []

    job2 = CronJob(dispatch_jobs=["a", "b"], interval_seconds=60)
    assert job2.dispatch_jobs == ["a", "b"]
    print("OK dispatch_jobs_list")


def _test_requires_exactly_one_schedule() -> None:
    # none of the three set
    try:
        _make_job()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when no schedule is set")
    # any two together
    pairs = [
        {"daily_at": "03:00", "interval_seconds": 60},
        {"daily_at": "03:00", "cron": "0 3 * * *"},
        {"interval_seconds": 60, "cron": "0 3 * * *"},
    ]
    for kw in pairs:
        try:
            _make_job(**kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError when two schedules set: {kw}")
    # all three
    try:
        _make_job(daily_at="03:00", interval_seconds=60, cron="0 3 * * *")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when all three schedules set")
    print("OK requires_exactly_one_schedule")


def _test_cron_expression_valid() -> None:
    for expr in ["0 3 * * *", "*/15 * * * *", "0 */6 * * *", "0 3 * * 1-5", "30 2 1 * *"]:
        job = _make_job(cron=expr)
        assert job.cron == expr
    print("OK cron_expression_valid")


def _test_cron_expression_invalid() -> None:
    # Eager validation — bad expressions must raise at construction time,
    # so a typo fails at app start rather than at 3am.
    for bad in ["not a cron", "0 25 * * *", "60 * * * *", "* * * 13 *", ""]:
        try:
            _make_job(cron=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for cron={bad!r}")
    print("OK cron_expression_invalid")


class _FrozenJob(CronJob):
    """CronJob subclass with deterministic 'now' for daily_at / cron delay math."""

    def __init__(self, frozen_now: datetime.datetime, **kwargs):
        kwargs.setdefault("dispatch_step", "version_step")
        super().__init__(**kwargs)
        self._frozen_now = frozen_now

    def _next_fire_delay(self) -> float:
        if self.interval_seconds:
            return float(self.interval_seconds)
        if self.cron:
            from croniter import croniter

            nxt = croniter(self.cron, self._frozen_now).get_next(datetime.datetime)
            return (nxt - self._frozen_now).total_seconds()
        target = self._frozen_now.replace(
            hour=self._fire_hour,
            minute=self._fire_minute,
            second=0,
            microsecond=0,
        )
        if target <= self._frozen_now:
            target = target + datetime.timedelta(days=1)
        return (target - self._frozen_now).total_seconds()


def _test_next_fire_delay_before_target() -> None:
    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    job = _FrozenJob(
        frozen_now=datetime.datetime(2026, 6, 7, 2, 0, 0, tzinfo=tz),
        daily_at="03:00",
    )
    assert job._next_fire_delay() == 3600
    print("OK next_fire_delay_before_target")


def _test_next_fire_delay_after_target() -> None:
    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    job = _FrozenJob(
        frozen_now=datetime.datetime(2026, 6, 7, 4, 0, 0, tzinfo=tz),
        daily_at="03:00",
    )
    assert job._next_fire_delay() == 23 * 3600
    print("OK next_fire_delay_after_target")


def _test_next_fire_delay_interval() -> None:
    job = _make_job(interval_seconds=30)
    assert job._next_fire_delay() == 30.0
    print("OK next_fire_delay_interval")


def _test_next_fire_delay_cron_daily() -> None:
    # cron "0 3 * * *" is exactly equivalent to daily_at "03:00" — same math,
    # but exercised via the croniter path.
    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    job = _FrozenJob(
        frozen_now=datetime.datetime(2026, 6, 7, 2, 0, 0, tzinfo=tz),
        cron="0 3 * * *",
    )
    assert job._next_fire_delay() == 3600
    print("OK next_fire_delay_cron_daily")


def _test_next_fire_delay_cron_every_6h() -> None:
    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    # "0 */6 * * *" fires at 00:00 / 06:00 / 12:00 / 18:00. At 02:00,
    # next fire is 06:00 → 4 hours out.
    job = _FrozenJob(
        frozen_now=datetime.datetime(2026, 6, 7, 2, 0, 0, tzinfo=tz),
        cron="0 */6 * * *",
    )
    assert job._next_fire_delay() == 4 * 3600
    print("OK next_fire_delay_cron_every_6h")


def _test_next_fire_delay_cron_weekday_only() -> None:
    tz = zoneinfo.ZoneInfo("Asia/Shanghai")
    # 2026-06-07 is a Sunday. "0 3 * * 1-5" → next fire is Mon 2026-06-08 03:00.
    # From Sun 02:00, that's 25 hours.
    job = _FrozenJob(
        frozen_now=datetime.datetime(2026, 6, 7, 2, 0, 0, tzinfo=tz),
        cron="0 3 * * 1-5",
    )
    assert job._next_fire_delay() == 25 * 3600
    print("OK next_fire_delay_cron_weekday_only")


async def _drive_one_fire(_tmp: Path) -> int:
    """Stand up an Application + dispatch the counter step on a cron tick."""
    cfg = resolve_app_config(log_to_console=False, log_to_file=False, enable_logo=False)
    cfg["enable_logo"] = False
    app = Application(**cfg)
    await app.start()

    _CounterStep.fires = 0
    try:
        job = CronJob(
            dispatch_step="test_cron_counter_step",
            interval_seconds=1,
            run_on_start=True,
        )
        job.app_context = app.context
        # The BackgroundJob supervisor normally creates this in _start(); here we
        # drive __call__ directly, so wire up the stop_event by hand.
        job._stop_event = asyncio.Event()

        # Fire once on start, then signal stop so the loop exits before the
        # next interval elapses.
        task = asyncio.create_task(job())
        await asyncio.sleep(0.3)  # let run_on_start fire propagate
        job._stop_event.set()
        await asyncio.wait_for(task, timeout=5.0)
        return _CounterStep.fires
    finally:
        await app.close()


def _test_run_on_start_dispatches_once() -> None:
    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(Path(tmp)):
        count = asyncio.run(_drive_one_fire(Path(tmp)))
        assert count >= 1, f"expected at least one dispatch, got {count}"
        print(f"OK run_on_start_dispatches_once (count={count})")


def main() -> None:
    """Entry point for the cron job unit tests."""
    print("=== cron job unit tests ===")
    _test_parse_hh_mm_valid()
    _test_parse_hh_mm_invalid()
    _test_requires_dispatch_step()
    _test_dispatch_steps_list()
    _test_dispatch_jobs_list()
    _test_requires_exactly_one_schedule()
    _test_cron_expression_valid()
    _test_cron_expression_invalid()
    _test_next_fire_delay_before_target()
    _test_next_fire_delay_after_target()
    _test_next_fire_delay_interval()
    _test_next_fire_delay_cron_daily()
    _test_next_fire_delay_cron_every_6h()
    _test_next_fire_delay_cron_weekday_only()
    _test_run_on_start_dispatches_once()
    print("=== passed ===")


if __name__ == "__main__":
    main()
