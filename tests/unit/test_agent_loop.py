"""One scheduler owns independent task and Chat workers through shutdown."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from reme.core.context import RuntimeContext
from reme.schema import Event, Response
from reme.steps.proactive.agent import ProactiveLoopStep, ProcessChatRemindersStep
from .test_proactive_runtime import running_manager
from .test_inbox_fork import prepare


@pytest.mark.asyncio
async def test_shared_loop_runs_both_consumers_and_releases_their_claims_on_stop(tmp_path):
    async with running_manager(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        await runtime.append_events(
            "monitor",
            [Event(id="next", event_ts=runtime.now(), source="test", description="New observation")],
        )
        task_started, chat_started, stop = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def execute(_agent, _events):
            task_started.set()
            await asyncio.Event().wait()

        async def reply(_hint, **_options):
            chat_started.set()
            await asyncio.Event().wait()

        async def run_job(name, **kwargs):
            if name == "run_proactive_agent":
                return Response(answer=await runtime.run_round(kwargs["agent_name"], execute))
            assert name == "process_chat_reminders"
            return await ProcessChatRemindersStep(app_context=context)(**kwargs)

        wrapper.reply = AsyncMock(side_effect=reply)
        step = ProactiveLoopStep(app_context=context)
        step.run_job = run_job
        loop = asyncio.create_task(step(RuntimeContext(stop_event=stop)))
        try:
            async with asyncio.timeout(3):
                await task_started.wait()
                await chat_started.wait()
                assert (await runtime.inbox.snapshot("monitor"))["in_flight_count"] == 1
                assert (await runtime.list_inputs("Remy", state="processing"))["total"] == 1
                stop.set()
                await loop
        finally:
            loop.cancel()
            await asyncio.gather(loop, return_exceptions=True)
        assert not runtime.workers and runtime.notifications.worker is None
        assert runtime.notifications.active_turn is None
        assert list(runtime.notifications.pending) == arguments["msg_ids"]
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == 1 and snapshot["in_flight_count"] == 0
        assert (await runtime.list_inputs("Remy", state="pending"))["total"] == 1


@pytest.mark.asyncio
async def test_shared_loop_requires_background_job_lifecycle(tmp_path):
    async with running_manager(tmp_path) as (_, context, _):
        with pytest.raises(RuntimeError, match="BackgroundJob stop_event"):
            await ProactiveLoopStep(app_context=context)()


@pytest.mark.parametrize("failed", ["task", "chat"])
@pytest.mark.asyncio
async def test_scheduler_failure_does_not_starve_the_other_consumer(tmp_path, monkeypatch, failed):
    async with running_manager(tmp_path) as (runtime, context, _):
        await prepare(runtime, context)
        started = asyncio.Event()
        expected = "process_chat_reminders" if failed == "task" else "run_proactive_agent"

        async def run_job(name, **_kwargs):
            assert name == expected
            started.set()
            return Response(answer={"skipped": "busy"})

        failure = AsyncMock(side_effect=OSError("scheduler unavailable"))
        if failed == "task":
            await runtime.create(
                {"agent_name": "timer", "task_description": "Report", "cron": "* * * * *"},
            )
            monkeypatch.setattr(runtime.runtimes["timer"], "_schedule_cron", failure)
        else:
            await runtime.append_events(
                "monitor",
                [Event(id="next", event_ts=runtime.now(), source="test", description="New observation")],
            )
            monkeypatch.setattr(runtime.notifications, "tick", failure)
        with pytest.raises(OSError, match="scheduler unavailable"):
            await runtime.tick(run_job)
        async with asyncio.timeout(1):
            await started.wait()


@pytest.mark.asyncio
async def test_manager_discovers_workspace_runtime_changes(tmp_path):
    async with running_manager(tmp_path) as (manager, _, _):
        name = "workspace_runtime"
        manager.registry.create({"agent_name": name, "task_description": "Follow changes"}, manager.now())
        run_job = AsyncMock()
        await manager.tick(run_job)
        assert manager.runtimes[name].agent_name == name
        manager.registry.delete(name, manager.now())
        await manager.tick(run_job)
        assert name not in manager.runtimes
        run_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_paused_runtime_cannot_fill_another_runtime_inbox(tmp_path):
    """Capacity for a paused runtime is isolated from the other runtimes."""
    async with running_manager(tmp_path) as (manager, _, _):
        await manager.create({"agent_name": "paused", "task_description": "Paused", "enabled": False})
        await manager.create({"agent_name": "work", "task_description": "Active"})
        manager.inbox.max_pending_items = 1
        event = Event(id="first", event_ts=manager.now(), source="test", description="Ready")
        await manager.append_events("paused", [event])
        async with asyncio.timeout(1):
            assert (await manager.append_events("work", [event]))["event_ids"] == [event.id]
        assert (await manager.status("paused"))["pending_events"] == 1


@pytest.mark.asyncio
async def test_invalid_runtime_definition_does_not_stop_healthy_worker(tmp_path):
    """Discovery preserves a corrupt definition and continues processing valid runtimes."""
    async with running_manager(tmp_path) as (manager, _, _):
        await manager.create({"agent_name": "broken", "task_description": "Broken config"})
        await manager.create({"agent_name": "work", "task_description": "Active"})
        config_path = manager.registry.path("broken") / "config.json"
        manager.files.write(config_path, "{bad json")
        await manager.append_events(
            "work",
            [Event(id="next", event_ts=manager.now(), source="test", description="Ready")],
        )

        async def run_job(name, agent_name):
            assert name == "run_proactive_agent" and agent_name == "work"
            execute = AsyncMock(return_value=(None, "session"))
            return Response(answer=await manager.run_round(agent_name, execute))

        await manager.tick(run_job)
        await manager.runtimes["work"].worker
        assert (await manager.status("work"))["pending_events"] == 0
        with pytest.raises(ValueError, match="Invalid Proactive Runtime broken"):
            await manager.status("broken")
        with pytest.raises(ValueError, match="Invalid Proactive Runtime already exists"):
            await manager.create({"agent_name": "broken", "task_description": "Do not overwrite"})
        assert config_path.read_text() == "{bad json"


@pytest.mark.asyncio
async def test_task_configuration_rejects_retired_fields_without_rewriting_user_files(tmp_path):
    async with running_manager(tmp_path) as (runtime, _, _):
        await runtime.create({"agent_name": "experiment", "task_description": "Keep original evidence"})
        path = runtime.registry.path("experiment") / "config.json"
        original = json.loads(path.read_text())
        invalid = {**original, "temporary": True}
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match="Extra inputs"):
            runtime.registry.get("experiment")
        assert json.loads(path.read_text()) == invalid
