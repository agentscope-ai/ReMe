"""Unit tests for the ``cron`` job."""

# pylint: disable=protected-access

import asyncio
from types import SimpleNamespace

from reme.components import R
from reme.components.job.base_job import BaseJob
from reme.components.job.cron_job import CronJob
from reme.steps.base_step import BaseStep


@R.register("test_cron_counter_step")
class _CounterStep(BaseStep):
    fires: int = 0

    async def execute(self):
        type(self).fires += 1
        self.context.response.success = True
        return self.context.response


def _test_cron_parameter_protocol() -> None:
    assert CronJob("* * * * *").cron_expr == "* * * * *"
    assert CronJob(cron="0 3 * * *").cron_expr == "0 3 * * *"
    print("OK cron_parameter_protocol")


async def _invalid_cron_raises() -> None:
    try:
        await CronJob(cron="not a cron")._start()
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid cron expression")


def _test_invalid_cron_raises_on_start() -> None:
    asyncio.run(_invalid_cron_raises())
    print("OK invalid_cron_raises_on_start")


async def _drive_steps_once() -> int:
    _CounterStep.fires = 0
    job = CronJob(
        cron="* * * * *",
        steps=[{"backend": "test_cron_counter_step"}],
    )
    job.app_context = SimpleNamespace(app_config=SimpleNamespace(language=""))
    await BaseJob._start(job)
    job._stop_event = asyncio.Event()
    job._next_fire_delay = lambda: 0.01

    task = asyncio.create_task(job())
    await asyncio.sleep(0.05)
    job._stop_event.set()
    await asyncio.wait_for(task, timeout=1)
    return _CounterStep.fires


def _test_executes_own_steps() -> None:
    count = asyncio.run(_drive_steps_once())
    assert count >= 1, f"expected cron to execute own steps, got {count}"
    print(f"OK executes_own_steps count={count}")


def main() -> None:
    """Run all tests."""
    print("=== cron job unit tests ===")
    _test_cron_parameter_protocol()
    _test_invalid_cron_raises_on_start()
    _test_executes_own_steps()
    print("=== passed ===")


if __name__ == "__main__":
    main()
