"""Tests for BaseJob and BackgroundJob."""

# pylint: disable=protected-access,missing-function-docstring,missing-class-docstring,no-self-argument,unused-argument

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from reme.enumeration import ComponentEnum
from reme.application import Application
from reme.components.base_component import BaseComponent
from reme.components.component_registry import ComponentRegistry
from reme.components.job.background_job import BackgroundJob
from reme.components.job.base_job import BaseJob
from reme.components.job.cron_job import CronJob
from reme.components.job.stream_job import StreamJob
from reme.components.job import cron_job as cron_job_module
from reme.schema import ComponentConfig


# -- helpers ------------------------------------------------------------------


def _make_registry_and_context(step_classes=None):
    """Build a fresh registry + minimal app_context for job tests."""
    reg = ComponentRegistry()
    if step_classes:
        for name, cls in step_classes.items():
            reg.register(cls, name)

    ctx = MagicMock()
    ctx.components = {}
    return reg, ctx


# -- BaseJob._resolve_step ---------------------------------------------------


def test_resolve_step_missing_backend():
    job = BaseJob(name="j")
    job.app_context = MagicMock()
    with pytest.raises(ValueError, match="missing the required 'backend'"):
        job._resolve_step(ComponentConfig(backend=""))


def test_resolve_step_unregistered_backend():
    job = BaseJob(name="j")
    job.app_context = MagicMock()
    with pytest.raises(ValueError, match="Unregistered backend"):
        job._resolve_step(ComponentConfig(backend="nonexistent_step"))


# -- BaseJob.__call__ error capture ------------------------------------------


def test_call_captures_exception():
    async def run():
        failing_step = AsyncMock(side_effect=RuntimeError("boom"))

        job = BaseJob(name="j")
        job.app_context = MagicMock()
        job.step_specs = []
        job._build_steps = lambda: [failing_step]

        response = await job()
        assert response.success is False
        assert "boom" in response.answer

    asyncio.run(run())


def test_call_runs_steps_in_order():
    async def run():
        call_order = []

        async def step1(ctx):
            call_order.append("s1")

        async def step2(ctx):
            call_order.append("s2")

        job = BaseJob(name="j")
        job.app_context = MagicMock()
        job.step_specs = []
        job._build_steps = lambda: [step1, step2]

        response = await job()
        assert response.success is True
        assert call_order == ["s1", "s2"]

    asyncio.run(run())


def test_base_job_merges_config_kwargs_into_context():
    async def run():
        seen = {}

        async def step(ctx):
            seen.update(ctx.data)

        job = BaseJob(name="j", default_value="from-config")
        job.app_context = MagicMock()
        job.step_specs = []
        job._build_steps = lambda: [step]

        response = await job(default_value="from-call", call_only=True)
        assert response.success is True
        assert seen == {"default_value": "from-call", "call_only": True}

    asyncio.run(run())


def test_stream_job_merges_config_kwargs_into_context():
    async def run():
        seen = {}

        async def step(ctx):
            seen.update(ctx.data)

        queue = asyncio.Queue()
        job = StreamJob(name="j", default_value="from-config")
        job.app_context = MagicMock()
        job.step_specs = []
        job._build_steps = lambda: [step]

        await job(stream_queue=queue)
        done = await queue.get()
        assert done.done is True
        assert seen["default_value"] == "from-config"

    asyncio.run(run())


# -- BaseJob._start requires app_context ------------------------------------


def test_start_without_app_context_raises():
    async def run():
        job = BaseJob(name="j")
        with pytest.raises(RuntimeError, match="app_context must be provided"):
            await job._start()

    asyncio.run(run())


# -- BackgroundJob._backoff_delay --------------------------------------------


def test_backoff_delay_increases():
    job = BackgroundJob(
        name="bg",
        backoff_base=1.0,
        backoff_cap=60.0,
    )
    delays = [job._backoff_delay(i) for i in range(10)]
    # Delay should generally increase (with jitter, so we check trend).
    assert delays[-1] >= delays[0] or delays[-1] == job.backoff_cap


def test_backoff_delay_capped():
    job = BackgroundJob(
        name="bg",
        backoff_base=1.0,
        backoff_cap=10.0,
    )
    for _ in range(100):
        delay = job._backoff_delay(20)
        assert delay <= job.backoff_cap


def test_backoff_delay_has_jitter():
    job = BackgroundJob(
        name="bg",
        backoff_base=1.0,
        backoff_cap=60.0,
    )
    delays = {job._backoff_delay(5) for _ in range(20)}
    assert len(delays) > 1


def test_backoff_delay_attempt_zero():
    job = BackgroundJob(
        name="bg",
        backoff_base=2.0,
        backoff_cap=60.0,
    )
    for _ in range(50):
        delay = job._backoff_delay(0)
        assert 0 < delay <= 2.0 * 1.5


# -- BackgroundJob supervisor loop -------------------------------------------


def test_supervisor_restarts_on_crash():
    async def run():
        call_count = 0
        stop = asyncio.Event()

        class CrashingJob(BackgroundJob):
            async def __call__(self_, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError("crash")
                stop.set()

        job = CrashingJob(
            name="bg",
            supervisor=True,
            backoff_base=0.01,
            backoff_cap=0.05,
        )
        job._stop_event = stop
        await job._run_with_supervisor()
        assert call_count == 3

    asyncio.run(run())


def test_supervisor_disabled_propagates_exception():
    async def run():
        class FatalJob(BackgroundJob):
            async def __call__(self_, **kwargs):
                raise RuntimeError("fatal")

        job = FatalJob(
            name="bg",
            supervisor=False,
        )
        job._stop_event = asyncio.Event()
        with pytest.raises(RuntimeError, match="fatal"):
            await job._run_with_supervisor()

    asyncio.run(run())


# -- BackgroundJob._wait_or_stop ----------------------------------------------


def test_wait_or_stop_returns_on_stop():
    async def run():
        job = BackgroundJob(name="bg")
        job._stop_event = asyncio.Event()
        job._stop_event.set()
        await job._wait_or_stop(10.0)

    asyncio.run(run())


# -- BackgroundJob._shutdown_task ---------------------------------------------


def test_shutdown_task_none():
    async def run():
        job = BackgroundJob(name="bg")
        job._task = None
        await job._shutdown_task()

    asyncio.run(run())


def test_shutdown_task_cancels_on_timeout():
    async def run():
        async def hang_forever():
            await asyncio.sleep(999)

        job = BackgroundJob(name="bg", close_timeout=0.05)
        job._task = asyncio.create_task(hang_forever())
        await job._shutdown_task()
        assert job._task is None

    asyncio.run(run())


def test_cron_uses_configured_timezone(monkeypatch):
    from zoneinfo import ZoneInfo

    seen = {}

    def zone_info(name):
        seen["timezone"] = name
        return ZoneInfo(name)

    monkeypatch.setattr(cron_job_module, "ZoneInfo", zone_info)

    job = CronJob(cron="0 0 * * *")
    job.app_context = SimpleNamespace(app_config=SimpleNamespace(timezone="America/New_York"))
    delay = job._next_fire_delay()

    assert delay > 0
    assert seen["timezone"] == "America/New_York"


def test_application_starts_jobs_base_stream_background_cron():
    async def run():
        order = []
        app = object.__new__(Application)
        app.context = SimpleNamespace(
            app_config=SimpleNamespace(thread_pool_max_workers=0),
            jobs={
                "cron": CronJob(cron="* * * * *", name="cron"),
                "background": BackgroundJob(name="background"),
                "stream": StreamJob(name="stream"),
                "base": BaseJob(name="base"),
            },
            thread_pool=None,
        )
        app._topological_order = lambda: []

        async def start_one(component):
            order.append(component.name)

        app._start_one = start_one

        await Application._start(app)
        assert order == ["base", "stream", "background", "cron"]

    asyncio.run(run())


def test_application_start_failure_propagates_and_closes_started_components():
    async def run():
        class GoodComponent(BaseComponent):
            component_type = ComponentEnum.TOKENIZER

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.closed = False

            async def _close(self):
                self.closed = True

        class BrokenComponent(BaseComponent):
            component_type = ComponentEnum.FILE_STORE

            async def _start(self):
                raise RuntimeError("boom")

        good = GoodComponent(name="good")
        bad = BrokenComponent(name="bad")
        app = object.__new__(Application)
        app.context = SimpleNamespace(
            app_config=SimpleNamespace(thread_pool_max_workers=0),
            jobs={},
            thread_pool=None,
        )
        app._started_components = []
        app._topological_order = lambda: [good, bad]
        app.logger = MagicMock()

        with pytest.raises(RuntimeError, match="boom"):
            await Application._start(app)

        assert good.closed is True
        assert not app._started_components

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== Job Tests ===")
    test_resolve_step_missing_backend()
    test_resolve_step_unregistered_backend()
    test_call_captures_exception()
    test_call_runs_steps_in_order()
    test_start_without_app_context_raises()
    test_backoff_delay_increases()
    test_backoff_delay_capped()
    test_backoff_delay_has_jitter()
    test_backoff_delay_attempt_zero()
    test_supervisor_restarts_on_crash()
    test_supervisor_disabled_propagates_exception()
    test_wait_or_stop_returns_on_stop()
    test_shutdown_task_none()
    test_shutdown_task_cancels_on_timeout()
    print("\n所有测试通过!")
