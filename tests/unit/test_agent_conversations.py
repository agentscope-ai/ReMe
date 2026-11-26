"""Service streams and request-owned subscriptions compose Runtime state."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from unittest.mock import AsyncMock

import pytest

from reme.core.context import RuntimeContext
from reme.schema import SessionMessage, SessionSnapshot, SessionStatus, StreamChunk
from reme.steps.proactive.api import AskAgentStep, AgentSessionStep, AgentViews
from reme.steps.proactive.chat import ChatAgentStep
from reme.steps.proactive.tools.manage_proactive_runtime import ManageProactiveRuntimeStep
from .proactive_helpers import running_system, queue_event


@pytest.mark.parametrize("entrypoint", ["chat", "query"])
@pytest.mark.parametrize("will_retry", [False, True])
@pytest.mark.asyncio
async def test_nonstream_execution_retains_provider_error_outcomes(tmp_path, monkeypatch, entrypoint, will_retry):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
        runtime = await manager.resolve("monitor")
        await runtime.ensure_session({})
        original = wrapper.reply_stream

        async def reply(*args, **kwargs):
            yield StreamChunk(chunk_type="error", chunk="Provider error", metadata={"will_retry": will_retry})
            async for chunk in original(*args, **kwargs):
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", reply)
        if entrypoint == "chat":
            response = await ChatAgentStep(app_context=context, agent_runtime="Remy")(query="Hello")
        else:
            response = await ManageProactiveRuntimeStep(app_context=context)(
                action="query",
                agent_name="monitor",
                question="Progress?",
            )
        assert response.success is will_retry


@pytest.mark.parametrize("agent_name", ["Remy", "monitor"])
@pytest.mark.parametrize(
    "status,stream_error",
    [("completed", False), ("failed", False), ("interrupted", False), ("waiting", False), ("completed", True)],
)
@pytest.mark.asyncio
async def test_conversation_preserves_outcome_metadata_inputs_and_one_transport(
    tmp_path,
    monkeypatch,
    agent_name,
    status,
    stream_error,
):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
        task = await manager.resolve("monitor")
        await queue_event(manager)
        if agent_name != "Remy":
            await task.ensure_session({})
        chunks = []

        async def reply(_query, **options):
            if stream_error:
                chunks.append(StreamChunk(chunk_type="error", chunk="Transport error"))
            chunks.extend(
                [
                    StreamChunk(chunk_type="content", chunk="Partial", turn_id="turn"),
                    StreamChunk(
                        chunk_type="reply_end",
                        status=status,
                        chunk={
                            "session_id": options["resume"],
                            "result": "Final",
                            "status": status,
                            "error": "Provider failed" if status == "failed" else None,
                            "raw": {"native": "retained"},
                            "usage": {"input_tokens": 3, "output_tokens": 2},
                            "structured_output": None,
                        },
                    ),
                ],
            )
            for chunk in chunks:
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", reply)
        request = RuntimeContext(stream_queue=asyncio.Queue())
        request.response.metadata["earlier"] = "retained"
        response = await AskAgentStep(app_context=context, agent_runtime="Remy")(
            request,
            agent_name=agent_name,
            query="Hello",
        )
        assert response is request.response and response.success == (status == "completed" and not stream_error)
        assert (
            response.answer == "Final"
            and response.metadata["earlier"] == "retained"
            and response.metadata["raw"] == {"native": "retained"}
        )
        assert response.metadata["events"] == [chunk.model_dump(mode="json") for chunk in chunks]
        emitted = list(request.stream_queue._queue)
        assert all(chunk.chunk_type == "data" for chunk in emitted) and emitted[-1].chunk["outcome"] == status
        assert "native" not in str(emitted[-1].chunk) and request.get("agent_name") == agent_name
        assert await task.inbox.pending_ids() == {"event-1"} and not task.tasks
        if agent_name == "Remy":
            assert context.components["agent_runtime"]["Remy"].activity.phase == status
        else:
            assert wrapper.created[-1]["fork_session"] and (await task.current_session()).session_id == "session-1"


@pytest.mark.parametrize("agent_name", ["Remy", "monitor", "timer"])
@pytest.mark.parametrize("completed", [True, False])
@pytest.mark.asyncio
async def test_only_latest_task_turn_is_projected_without_mutating_native_history(tmp_path, agent_name, completed):
    async with running_system(
        tmp_path,
        runtimes=[{"agent_name": "monitor"}, {"agent_name": "timer", "cron": "0 8 * * *"}],
    ) as (manager, context, wrapper):
        runtime = (
            context.components["agent_runtime"]["Remy"] if agent_name == "Remy" else await manager.resolve(agent_name)
        )
        sid = await runtime.ensure_session({})
        snapshot = SessionSnapshot(
            session_id=sid,
            last_turn_id="latest",
            state=SessionStatus(session_id=sid, status="idle" if completed else "active"),
            messages=[
                SessionMessage(id="old-user", role="user", content="Old", turn_id="old"),
                SessionMessage(id="old-answer", role="assistant", content="Old answer", turn_id="old"),
                SessionMessage(id="latest-user", role="user", content="Latest", turn_id="latest"),
                SessionMessage(
                    id="latest-answer",
                    role="assistant",
                    content="Result",
                    turn_id="latest",
                    completed=completed,
                    chunks=[StreamChunk(chunk_type="content", block_id="answer", chunk="Result")],
                ),
            ],
        )
        original = snapshot.model_dump()
        wrapper.read_session = AsyncMock(return_value=snapshot)
        value = (await AgentSessionStep(app_context=context, agent_runtime="Remy")(agent_name=agent_name)).answer
        assert [row["id"] for row in value["messages"]] == (
            ["old-user", "old:assistant", "latest-user", "latest:assistant"]
            if agent_name == "Remy"
            else ["latest-user", "latest:assistant"]
        )
        assert value["messages"][-1]["completed"] is completed and snapshot.model_dump() == original
        assert (await runtime.current_session()).session_id == sid


@pytest.mark.asyncio
async def test_subscriptions_replace_turn_and_session_and_release_listeners(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}]) as (manager, context, wrapper):
        runtime = await manager.resolve("monitor")
        sid = await runtime.ensure_session({})
        snapshot = SessionSnapshot(
            session_id=sid,
            last_turn_id="one",
            state=SessionStatus(session_id=sid, status="idle"),
            messages=[SessionMessage(id="input-1", role="user", content="First", turn_id="one")],
        )
        wrapper.read_session = AsyncMock(side_effect=lambda _: snapshot)
        views = AgentViews(AgentSessionStep(app_context=context, agent_runtime="Remy"))
        events = views.events("monitor", interval=0.01)
        try:
            assert (await anext(events))["type"] == "snapshot"
            snapshot.last_turn_id = "two"
            snapshot.messages.append(SessionMessage(id="input-2", role="user", content="Second", turn_id="two"))
            changed = await asyncio.wait_for(anext(events), 1)
            assert changed["type"] == "snapshot" and [row["id"] for row in changed["messages"]] == ["input-2"]
            next_sid = await runtime.ensure_session({}, new=True)
            snapshot.session_id = next_sid
            snapshot.state.session_id = next_sid
            assert (await asyncio.wait_for(anext(events), 1))["agent"]["session_id"] == next_sid
        finally:
            await events.aclose()
        assert not runtime.live_turns.listeners


@pytest.mark.asyncio
async def test_remy_subscription_survives_damaged_task_history_and_definition(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor"}, {"agent_name": "broken"}]) as (
        manager,
        context,
        _,
    ):
        (manager.registry.path("monitor") / "sessions.jsonl").write_text("invalid record\n")
        broken = (await manager.resolve("broken")).config_path
        broken.write_text("invalid definition\n")
        views = AgentViews(AgentSessionStep(app_context=context, agent_runtime="Remy"))
        events = views.events("Remy")
        try:
            snapshot = await asyncio.wait_for(anext(events), 1)
            values = {item["agent_name"]: item for item in snapshot["agents"]}
            assert (
                values["Remy"]["status"] == "idle"
                and values["monitor"]["status"] == values["broken"]["status"] == "error"
            )
            assert broken.read_text() == "invalid definition\n"
        finally:
            await events.aclose()
