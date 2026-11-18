"""All backends share active snapshots and P0 delivery; P1/P2 retain idle scheduling."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from unittest.mock import AsyncMock

import pytest

from reme.schema import SessionSnapshot, SessionStatus
from reme.schema.proactive_agent import ProactiveRunResult
from reme.components import RuntimeContext
from reme.steps.proactive.agent import AgentChatStep, InboxChatStep, ProactiveQueryStep
from reme.steps.proactive.tools.inbox import ReadInboxStep, SendChatMessageStep
from .test_agent_runtime import create_agent, running_runtime
from .test_inbox_fork import append_message, prepare


@pytest.mark.asyncio
@pytest.mark.parametrize("step_class", [AgentChatStep, ProactiveQueryStep])
async def test_query_forks_latest_running_session_instead_of_previous_receipt(tmp_path, step_class):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        previous = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now())
        await runtime.run(
            "monitor",
            AsyncMock(return_value=(ProactiveRunResult(notifications=[]), previous)),
        )
        await runtime.append("monitor", {"event_ts": runtime.now(), "source": "test", "description": "New"})
        entered, release = asyncio.Event(), asyncio.Event()
        source = None

        async def execute(_agent, _events):
            nonlocal source
            source = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now(), new=True)
            entered.set()
            await release.wait()
            return ProactiveRunResult(notifications=[]), source

        running = asyncio.create_task(runtime.run("monitor", execute))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            response = await asyncio.wait_for(
                step_class(app_context=context)(
                    RuntimeContext(stream_queue=asyncio.Queue()),
                    agent_name="monitor",
                    query="Current progress?",
                ),
                1,
            )
            assert response.success
            assert wrapper.created[-1]["resume"] == source != previous
            assert (await runtime.sessions.current("monitor")).session_id == source
            assert not running.done()
            assert (await runtime.inbox.snapshot("monitor"))["in_flight_count"] == 1
            assert wrapper.turns[-1][0] == "Current progress?"
            assert "monitor" not in runtime.live_turns.turns
        finally:
            release.set()
            await running


@pytest.mark.asyncio
async def test_p0_forks_and_appends_while_foreground_chat_keeps_running(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context, priority="P0")
        source = (await runtime.sessions.current("Remy")).session_id
        wrapper.get_session_status = AsyncMock(return_value=SessionStatus(session_id=source, status="active"))
        wrapper.read_session = AsyncMock(
            return_value=SessionSnapshot(
                session_id=source,
                state=SessionStatus(session_id=source, status="active"),
                last_turn_id="active-turn",
            ),
        )
        entered, release = asyncio.Event(), asyncio.Event()

        async def foreground():
            async with runtime.chat_turn():
                entered.set()
                await release.wait()

        task = asyncio.create_task(foreground())
        await asyncio.wait_for(entered.wait(), 1)

        async def append(session_id, content, **kwargs):
            assert runtime.chat_lock.locked() and not task.done()
            assert kwargs["delivery"] == "immediate"
            assert kwargs["expected_turn_id"] == "active-turn"
            assert kwargs["expected_revision"] is None
            return append_message(session_id, content, **kwargs)

        async def reply(_hint, **options):
            assert options["resume"] == source and options["fork_session"]
            bound = options["injected_job_kwargs"]
            await ReadInboxStep(app_context=context)(**bound)
            await SendChatMessageStep(app_context=context)(content="Urgent update", **bound)
            return {"status": "completed", "result": ""}

        wrapper.append_session = AsyncMock(side_effect=append)
        wrapper.reply = AsyncMock(side_effect=reply)
        try:
            response = await asyncio.wait_for(InboxChatStep(app_context=context)(**arguments), 1)
            assert response.answer == {"sent": True}
            assert not task.done() and not runtime.notifications.pending
            assert (await runtime.sessions.current("Remy")).session_id == source
        finally:
            release.set()
            await task


@pytest.mark.parametrize("priority", ["P1", "P2"])
@pytest.mark.asyncio
async def test_nonurgent_forks_keep_waiting_for_chat_idle(tmp_path, priority):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context, priority=priority)
        wrapper.reply = AsyncMock()
        async with runtime.chat_lock:
            response = await asyncio.wait_for(InboxChatStep(app_context=context)(**arguments), 1)
        assert response.answer == {"skipped": "busy"}
        wrapper.reply.assert_not_awaited()
        assert list(runtime.notifications.pending) == arguments["msg_ids"]


@pytest.mark.asyncio
async def test_p0_defers_when_a_new_user_turn_is_already_queued(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        arguments = await prepare(runtime, context, priority="P0")
        source = (await runtime.sessions.current("Remy")).session_id
        wrapper.get_session_status = AsyncMock(return_value=SessionStatus(session_id=source, status="active"))
        wrapper.reply = AsyncMock()
        entered, release = asyncio.Event(), asyncio.Event()

        async def foreground():
            async with runtime.chat_turn():
                entered.set()
                await release.wait()

        first = asyncio.create_task(foreground())
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(foreground())
        try:
            await asyncio.sleep(0)
            assert len(runtime.chat_tasks) == 2
            response = await asyncio.wait_for(InboxChatStep(app_context=context)(**arguments), 1)
            assert response.answer == {"skipped": "busy"}
            wrapper.reply.assert_not_awaited()
        finally:
            release.set()
            await asyncio.gather(first, second)
