"""Foreground leases and urgent native delivery are independent."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio

import pytest

from reme.steps.proactive.api import AskAgentStep
from reme.steps.proactive.chat import ProcessChatNotificationsStep
from reme.steps.proactive.tools import SubmitChatReplyStep, ManageProactiveRuntimeStep
from reme.core.context import RuntimeContext
from .proactive_helpers import running_system, emit_notification


@pytest.mark.parametrize("priority", ["P0", "P1", "P2"])
@pytest.mark.asyncio
async def test_urgent_delivery_can_reach_an_active_foreground_turn(tmp_path, monkeypatch, priority):
    async with running_system(tmp_path) as (manager, context, wrapper):
        ref = await emit_notification(manager, context, priority=priority)
        chat = context.components["agent_runtime"]["Remy"]
        session = await chat.ensure_session({})
        wrapper.states[session] = "active"
        read = wrapper.read_session

        async def active(sid):
            value = await read(sid)
            return value.model_copy(update={"last_turn_id": "active-turn"})

        monkeypatch.setattr(wrapper, "read_session", active)
        # Keep the formal Chat signature stable while preparing this fork.
        from reme.steps.proactive.chat.chat import ChatAgentStep

        step = ChatAgentStep(app_context=context, agent_runtime="Remy")
        step.context = RuntimeContext()
        session = await chat.ensure_session(chat.session_options())
        wrapper.states[session] = "active"
        entered, release = asyncio.Event(), asyncio.Event()

        async def foreground():
            async with chat.chat_turn():
                entered.set()
                await release.wait()

        task = asyncio.create_task(foreground())
        await entered.wait()

        async def reply(_query, **options):
            return await SubmitChatReplyStep(app_context=context, agent_runtime="Remy")(
                action="send",
                content="Urgent",
                **options["injected_job_kwargs"],
            )

        monkeypatch.setattr(wrapper, "reply", reply)
        try:
            response = await asyncio.wait_for(
                ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")(
                    priority=priority,
                    msg_ids=[ref.id],
                ),
                1,
            )
            if priority == "P0":
                assert response.answer["sent"] and len(wrapper.messages[session]) == 1
            else:
                assert response.answer == {"skipped": "busy"} and ref.id in await chat.inbox.pending_ids()
            assert not task.done()
        finally:
            release.set()
            await task


@pytest.mark.parametrize("kind", ["service", "tool"])
@pytest.mark.asyncio
async def test_readonly_query_forks_latest_running_task_session_without_consuming_inputs(tmp_path, kind):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        wrapper,
    ):
        runtime = await manager.resolve("monitor")
        previous = await runtime.ensure_session({})
        entered, release = asyncio.Event(), asyncio.Event()

        async def work():
            async with runtime.execution():
                await runtime.ensure_session({}, new=True)
                entered.set()
                await release.wait()

        task = asyncio.create_task(work())
        await entered.wait()
        source = (await runtime.current_session()).session_id
        try:
            step = (
                AskAgentStep(app_context=context, agent_runtime="Remy")
                if kind == "service"
                else ManageProactiveRuntimeStep(app_context=context)
            )
            values = {"query": "Progress?"} if kind == "service" else {"action": "query", "question": "Progress?"}
            response = await step(RuntimeContext(stream_queue=asyncio.Queue()), agent_name="monitor", **values)
            assert response.success and wrapper.created[-1]["resume"] == source != previous
            assert wrapper.created[-1]["sandbox"] == "read-only" and wrapper.created[-1]["skills"] == []
            assert (await runtime.current_session()).session_id == source and not task.done()
            assert wrapper.released[-1] != source
        finally:
            release.set()
            await task


@pytest.mark.parametrize("arrival", ["before_snapshot", "during_snapshot", "during_eligibility"])
@pytest.mark.asyncio
async def test_p0_does_not_overtake_a_queued_foreground_request(tmp_path, monkeypatch, arrival):
    async with running_system(tmp_path) as (manager, context, wrapper):
        ref = await emit_notification(manager, context, priority="P0")
        chat = context.components["agent_runtime"]["Remy"]
        session = await chat.ensure_session({})
        wrapper.states[session] = "active"
        entered, release = asyncio.Event(), asyncio.Event()

        async def foreground():
            async with chat.chat_turn():
                entered.set()
                await release.wait()

        active = asyncio.create_task(foreground())
        await entered.wait()
        queued = None

        async def queue_request():
            nonlocal queued
            queued = asyncio.create_task(foreground())
            await asyncio.sleep(0)

        read = wrapper.read_session

        async def snapshot(identifier):
            value = await read(identifier)
            if arrival == "during_snapshot" and queued is None:
                await queue_request()
            return value.model_copy(update={"last_turn_id": "active-turn"})

        async def eligible(_items):
            if arrival == "during_eligibility" and queued is None:
                await queue_request()
            return True

        monkeypatch.setattr(wrapper, "read_session", snapshot)
        if arrival == "before_snapshot":
            await queue_request()
        notifications = ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")
        try:
            with pytest.raises(RuntimeError, match="foreground|conversation changed"):
                async with chat.notification_turn(
                    session,
                    "P0",
                    [ref.id],
                    format_item=notifications.format_item,
                    eligible=eligible,
                ) as turn:
                    await chat.submit_reply(turn.id, "send", "Old reminder")
            assert ref.id in await chat.inbox.pending_ids()
            assert not wrapper.messages[session]
        finally:
            release.set()
            await active
            if queued is not None:
                await queued
