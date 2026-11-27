"""Task scheduling, schedule queue pressure, cooldown and frozen retry batches."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.core.context import RuntimeContext
from reme.schema import Event, Response
from reme.steps.proactive.loop import AgentManagerLoopStep
from .proactive_helpers import running_system, queue_event


@pytest.mark.asyncio
async def test_full_time_queue_still_starts_its_consumer(tmp_path, monkeypatch):
    async with running_system(
        tmp_path,
        runtimes=[{"agent_type": "time", "agent_name": "clock", "schedule": "0 * * * *"}],
    ) as (
        manager,
        context,
        _,
    ):
        runtime = await manager.resolve("clock")
        runtime.inbox.max_pending_items = 1
        definition = await runtime.definition()
        trigger = runtime.next_trigger(definition)
        monkeypatch.setattr(runtime, "now", lambda: trigger)
        await runtime.inbox.append(
            Event(id="old", event_ts=trigger - timedelta(hours=1), source="time", description="Backlog"),
        )
        called = AsyncMock(return_value=Response())
        loop = AgentManagerLoopStep(app_context=context)
        loop.context = RuntimeContext()
        monkeypatch.setattr(loop, "run_job", called)
        await loop.tick_runtime(runtime, definition)
        await runtime.worker
        called.assert_awaited_once_with(runtime.run_job, agent_name="clock")
        assert runtime.schedule[1] == trigger
        batch = await runtime.inbox.claim()
        await runtime.inbox.ack(batch["token"])
        await loop.tick_runtime(runtime, definition)
        await runtime.worker
        assert (await runtime.inbox.read())["items"][0]["item"]["id"] == f"time-clock-{int(trigger.timestamp())}"
        assert runtime.schedule[1] > trigger


@pytest.mark.asyncio
async def test_pausing_does_not_cancel_current_lease_and_resuming_keeps_inputs(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        _,
    ):
        await queue_event(manager)
        runtime = await manager.resolve("monitor")
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow(*_, **__):
            entered.set()
            await release.wait()
            return Response()

        loop = AgentManagerLoopStep(app_context=context)
        loop.context = RuntimeContext()
        monkeypatch.setattr(loop, "run_job", slow)
        await loop.tick()
        await entered.wait()
        worker = runtime.worker
        await manager.update("monitor", {"enabled": False})
        await loop.tick()
        assert runtime.worker is worker and not worker.done()
        release.set()
        await worker
        await loop.tick()
        assert runtime.worker is None
        assert await runtime.inbox.pending_ids() == {"event-1"}


@pytest.mark.asyncio
async def test_cooldown_uses_last_durable_receipt(tmp_path, monkeypatch):
    async with running_system(
        tmp_path,
        runtimes=[{"agent_type": "event", "agent_name": "monitor", "min_trigger_interval_seconds": 60}],
    ) as (
        manager,
        context,
        _,
    ):
        runtime = await manager.resolve("monitor")
        await queue_event(manager)
        batch = await runtime.inbox.claim()
        await runtime.inbox.ack(batch["token"])
        receipt = (await runtime.inbox.snapshot())["receipt"]
        await queue_event(manager, identifier="next")
        now = receipt.consume_ts + timedelta(seconds=59)
        monkeypatch.setattr(runtime, "now", lambda: now)
        loop = AgentManagerLoopStep(app_context=context)
        loop.context = RuntimeContext()
        call = AsyncMock(return_value=Response())
        monkeypatch.setattr(loop, "run_job", call)
        await loop.tick()
        assert runtime.worker is None
        now += timedelta(seconds=1)
        await loop.tick()
        await runtime.worker
        assert call.await_count == 1


@pytest.mark.asyncio
async def test_failed_batch_does_not_grow_when_later_inputs_arrive(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        from .test_proactive_run import prepare_round, install_round, SKIP

        runtime, record = await prepare_round(manager, context, monkeypatch)
        await queue_event(manager)
        install_round(wrapper, runtime, record, monkeypatch, status="failed")
        assert not (await context.jobs[runtime.run_job](agent_name="monitor")).success
        await queue_event(manager, identifier="later")
        assert runtime.retry_items == ["event-1"]
        install_round(wrapper, runtime, record, monkeypatch, text=SKIP)
        assert (await context.jobs[runtime.run_job](agent_name="monitor")).success
        assert await runtime.inbox.pending_ids() == {"later"}


@pytest.mark.asyncio
async def test_shutdown_joins_owned_task_workers(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        _,
    ):
        await queue_event(manager)
        entered = asyncio.Event()

        async def slow(*_, **__):
            entered.set()
            await asyncio.Event().wait()

        loop = AgentManagerLoopStep(app_context=context)
        monkeypatch.setattr(loop, "run_job", slow)
        stop = asyncio.Event()
        task = asyncio.create_task(loop(RuntimeContext(stop_event=stop)))
        await asyncio.wait_for(entered.wait(), 1)
        worker = manager.runtimes["monitor"].worker
        stop.set()
        await asyncio.wait_for(task, 2)
        assert worker.cancelled()
