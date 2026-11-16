"""Formal conversation steps preserve stream outcomes and shared request state."""

# pylint: disable=missing-function-docstring

import asyncio
from unittest.mock import AsyncMock

import pytest

from reme.components import RuntimeContext
from reme.enumeration import ChunkEnum
from reme.schema import SessionMessage, SessionSnapshot, SessionStatus, StreamChunk
from reme.components.agent_runtime._internal.views import AgentViews
from reme.steps.proactive.agent import AgentChatStep, ChatAgentStep
from .test_agent_runtime import create_agent, running_runtime


@pytest.mark.parametrize(
    "step_class,agent_id",
    [(ChatAgentStep, "chat"), (AgentChatStep, "chat"), (AgentChatStep, "monitor")],
)
@pytest.mark.parametrize(
    "status,stream_error",
    [("completed", False), ("failed", False), ("interrupted", False), ("waiting", False), ("completed", True)],
)
@pytest.mark.asyncio
async def test_conversation_streams_preserve_outcomes_metadata_and_caller_inputs(
    tmp_path,
    monkeypatch,
    step_class,
    agent_id,
    status,
    stream_error,
):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        await create_agent(runtime)
        chunks = []

        async def reply_stream(_query, **options):
            if stream_error:
                chunks.append(StreamChunk(chunk_type=ChunkEnum.ERROR, chunk="transport error"))
            chunks.extend(
                [
                    StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="partial", turn_id="turn"),
                    StreamChunk(
                        chunk_type=ChunkEnum.REPLY_END,
                        status=status,
                        chunk={
                            "session_id": options["resume"],
                            "result": "final answer",
                            "structured_output": None,
                            "usage": {"input_tokens": 3, "output_tokens": 2},
                            "status": status,
                            "error": "provider failed" if status == "failed" else None,
                            "raw": {"native": "retained"},
                        },
                    ),
                ],
            )
            for chunk in chunks:
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", reply_stream)
        request = RuntimeContext(stream_queue=asyncio.Queue())
        request.response.metadata["previous_step"] = "retained"
        arguments = {} if step_class is ChatAgentStep else {"agent_id": agent_id}
        response = await step_class(app_context=context)(request, query="Hello", **arguments)

        assert response is request.response
        assert response.success is (status == "completed" and not stream_error)
        assert response.answer == "final answer"
        assert response.metadata["agent_id"] == agent_id
        assert response.metadata["status"] == status
        assert response.metadata["raw"] == {"native": "retained"}
        assert response.metadata["usage"] == {"input_tokens": 3, "output_tokens": 2}
        assert response.metadata["previous_step"] == "retained"
        assert response.metadata["events"] == [chunk.model_dump(mode="json") for chunk in chunks]
        assert [request.stream_queue.get_nowait() for _ in chunks] == chunks
        assert request.get("agent_id") == arguments.get("agent_id")
        assert not runtime.chat_lock.locked() and not runtime.agent_tasks
        assert (await runtime.status("monitor"))["pending_events"] == 1
        activity = runtime.activities[agent_id]
        assert activity.turn_id == "turn" and activity.phase == status


@pytest.mark.asyncio
async def test_subscription_reports_portable_tool_activity_and_cancellation(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        entered = asyncio.Event()

        async def stream(_query, **_options):
            yield StreamChunk(
                chunk_type=ChunkEnum.TOOL_CALL,
                chunk="read memory",
                tool_call_id="call",
                tool_call_name="read_file",
                turn_id="turn",
            )
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(wrapper, "reply_stream", stream)
        monkeypatch.setattr(
            wrapper,
            "read_session",
            AsyncMock(
                side_effect=lambda sid: SessionSnapshot(
                    session_id=sid,
                    state=SessionStatus(session_id=sid, status="idle"),
                ),
            ),
        )
        task = asyncio.create_task(
            AgentChatStep(app_context=context)(RuntimeContext(stream_queue=asyncio.Queue()), query="Hello"),
        )
        try:
            await entered.wait()
            events = AgentViews(runtime).events("chat", interval=0)
            try:
                value = await anext(events)
                activity = value["activities"]["chat"]
                assert activity["phase"] == "working" and activity["tool"] == "read_file"
                assert activity["turn_id"] == "turn" and value["status"] == "active"
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                value = await anext(events)
                assert value["activities"]["chat"]["phase"] == "interrupted"
                assert value["status"] == "idle"
            finally:
                await events.aclose()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("agent_id", ["chat", "monitor", "timer"])
@pytest.mark.parametrize("completed", [True, False])
@pytest.mark.asyncio
async def test_task_views_keep_only_latest_turn_without_mutating_native_history(tmp_path, agent_id, completed):
    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        await create_agent(runtime)
        await runtime.create(
            {"name": "timer", "description": "Daily report", "trigger_type": "cron", "cron": "0 8 * * *"},
        )
        sid = await runtime.sessions.ensure(agent_id, wrapper, {}, runtime.now())
        snapshot = SessionSnapshot(
            session_id=sid,
            last_turn_id="latest",
            state=SessionStatus(session_id=sid, status="idle" if completed else "active"),
            messages=[
                SessionMessage(id="old-user", role="user", content="Old input", turn_id="old"),
                SessionMessage(id="old-answer", role="assistant", content="Old result", turn_id="old"),
                SessionMessage(id="latest-user", role="user", content="Latest input", turn_id="latest"),
                SessionMessage(id="latest-progress", role="assistant", content="Checking sources", turn_id="latest"),
                SessionMessage(
                    id="latest-answer",
                    role="assistant",
                    content="Latest result",
                    turn_id="latest",
                    completed=completed,
                    chunks=[StreamChunk(chunk_type=ChunkEnum.CONTENT, block_id="answer", chunk="Latest result")],
                ),
            ],
        )
        original = snapshot.model_dump()
        wrapper.read_session = AsyncMock(return_value=snapshot)
        value = await runtime.read_agent(agent_id)
        expected = snapshot.messages if agent_id == "chat" else snapshot.messages[2:]
        assert [message["id"] for message in value["snapshot"]["messages"]] == [message.id for message in expected]
        assert value["snapshot"]["messages"][-1]["completed"] is completed
        assert value["snapshot"]["messages"][-1]["chunks"][0]["block_id"] == "answer"
        assert snapshot.model_dump() == original
        wrapper.read_session.assert_awaited_once_with(sid)
        assert (await runtime.sessions.current(agent_id)).session_id == sid


@pytest.mark.parametrize("trigger_type", ["event", "cron"])
@pytest.mark.asyncio
async def test_task_subscription_atomically_replaces_turns_and_reconnects_without_old_messages(tmp_path, trigger_type):
    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        await runtime.create(
            {
                "name": "monitor",
                "description": "Report",
                "trigger_type": trigger_type,
                **({"cron": "0 8 * * *"} if trigger_type == "cron" else {}),
            },
        )
        sid = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now())
        snapshot = SessionSnapshot(
            session_id=sid,
            state=SessionStatus(session_id=sid, status="idle"),
            last_turn_id="old",
            messages=[SessionMessage(id="old", role="assistant", content="Old result", turn_id="old")],
        )
        wrapper.read_session = AsyncMock(return_value=snapshot)
        views = AgentViews(runtime)
        events = views.events("monitor", interval=0, heartbeat=0)
        try:
            assert (await anext(events))["snapshot"]["messages"][0]["id"] == "old"
            snapshot.last_turn_id = "new"
            empty = await anext(events)
            assert empty["type"] == "snapshot" and empty["snapshot"]["messages"] == []
            snapshot.messages.extend(
                [
                    SessionMessage(id="new-user", role="user", content="New input", turn_id="new"),
                    SessionMessage(
                        id="new-answer",
                        role="assistant",
                        content="Working",
                        turn_id="new",
                        completed=False,
                    ),
                ],
            )
            latest = await anext(events)
            assert latest["type"] == "snapshot"
            assert [message["id"] for message in latest["snapshot"]["messages"]] == ["new-user", "new-answer"]
            assert (await anext(events))["type"] == "heartbeat"
            reconnected = views.events("monitor", interval=0)
            try:
                assert (await anext(reconnected)) == latest
            finally:
                await reconnected.aclose()
            next_sid = await runtime.sessions.ensure("monitor", wrapper, {}, runtime.now(), new=True)
            snapshot.session_id = next_sid
            snapshot.state.session_id = next_sid
            changed = await anext(events)
            assert changed["type"] == "snapshot" and changed["session_id"] == next_sid
        finally:
            await events.aclose()
