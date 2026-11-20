"""One scheduler owns independent task and Chat workers through shutdown."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from unittest.mock import AsyncMock

import pytest

from reme.core.context import RuntimeContext
from reme.schema import Response
from reme.steps.proactive.agent import AgentLoopStep, InboxChatStep
from reme.steps.proactive.tools.inbox import ReadInboxStep
from .test_agent_runtime import running_runtime
from .test_inbox_fork import prepare


@pytest.mark.asyncio
async def test_shared_loop_runs_both_consumers_and_releases_their_claims_on_stop(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context)
        await runtime.append(
            "monitor",
            {"id": "next", "event_ts": runtime.now(), "source": "test", "description": "New observation"},
        )
        task_started, chat_started, stop = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def execute(_agent, _events):
            task_started.set()
            await asyncio.Event().wait()

        async def reply(_hint, **options):
            await ReadInboxStep(app_context=context)(**options["injected_job_kwargs"])
            chat_started.set()
            await asyncio.Event().wait()

        async def run_job(name, **kwargs):
            if name == "run_proactive_agent":
                return Response(answer=await runtime.run(kwargs["agent_name"], execute))
            assert name == "process_chat_reminders"
            return await InboxChatStep(app_context=context)(**kwargs)

        wrapper.reply = AsyncMock(side_effect=reply)
        step = AgentLoopStep(app_context=context)
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
    async with running_runtime(tmp_path) as (_, context, _):
        with pytest.raises(RuntimeError, match="BackgroundJob stop_event"):
            await AgentLoopStep(app_context=context)()


@pytest.mark.parametrize("failed", ["task", "chat"])
@pytest.mark.asyncio
async def test_scheduler_failure_does_not_starve_the_other_consumer(tmp_path, monkeypatch, failed):
    async with running_runtime(tmp_path) as (runtime, context, _):
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
            monkeypatch.setattr(runtime, "_schedule_cron", failure)
        else:
            await runtime.append(
                "monitor",
                {"id": "next", "event_ts": runtime.now(), "source": "test", "description": "New observation"},
            )
            monkeypatch.setattr(runtime.notifications, "tick", failure)
        with pytest.raises(OSError, match="scheduler unavailable"):
            await runtime.tick(run_job)
        async with asyncio.timeout(1):
            await started.wait()
