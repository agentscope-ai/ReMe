"""Regression coverage for bounded retries and durable chat read acknowledgements."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.components import RuntimeContext
from reme.schema import Event, Response
from reme.schema.proactive_agent import ProactiveAgentConfig, ProactiveRunResult
from reme.steps.proactive.agent import AgentChatStep, ProactiveAgentStep
from reme.steps.proactive.tools.append_event import AppendEventStep
from .test_agent_runtime import completed_result, create_agent, running_runtime


@pytest.mark.asyncio
async def test_failed_batch_does_not_grow_when_new_observations_arrive(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.update("monitor", {"max_batch_events": 1})
        with pytest.raises(TimeoutError):
            await runtime.run("monitor", AsyncMock(side_effect=TimeoutError))
        await runtime.append(
            "monitor",
            {"id": "event-2", "event_ts": runtime.now(), "source": "test", "description": "New"},
        )
        execute = AsyncMock(return_value=completed_result(runtime))
        result = await runtime.run("monitor", execute)
        assert result["processed"] == 1
        assert [item["id"] for item in execute.call_args.args[1]] == ["event-1"]
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1
        assert runtime.executions["monitor"].retry_items is None


@pytest.mark.asyncio
async def test_append_event_rechecks_trigger_after_a_concurrent_update(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        append = runtime.append

        async def change_trigger_then_append(agent_id, events):
            await runtime.update(agent_id, {"trigger_type": "cron", "cron": "0 * * * *"})
            return await append(agent_id, events)

        monkeypatch.setattr(runtime, "append", change_trigger_then_append)
        with pytest.raises(ValueError, match="external observations require an event agent"):
            await AppendEventStep(app_context=context)(
                agent_id="monitor",
                events=[{"id": "external", "event_ts": runtime.now(), "source": "test", "description": "Next"}],
            )
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.asyncio
async def test_runtime_append_rejects_the_reserved_cron_source(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        with pytest.raises(ValueError, match="reserved framework source"):
            await runtime.append(
                "monitor",
                {"id": "external", "event_ts": runtime.now(), "source": "cron", "description": "Forged trigger"},
            )
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 1


@pytest.mark.parametrize("size", [0, -1, True, "2", 1.5])
def test_event_batch_size_rejects_invalid_values(size):
    with pytest.raises(ValueError):
        ProactiveAgentConfig(name="monitor", trigger_type="event", max_batch_events=size)


@pytest.mark.parametrize("trigger_type", ["event", "cron"])
@pytest.mark.asyncio
async def test_automatic_rounds_start_fresh_sessions_while_user_followups_resume(tmp_path, monkeypatch, trigger_type):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        values = {"name": "monitor", "description": "Monitor", "trigger_type": trigger_type}
        if trigger_type == "cron":
            values["cron"] = "0 * * * *"
        await runtime.create(values)
        summary = tmp_path / "daily" / runtime.now().date().isoformat() / "monitor.md"
        original_reply = wrapper.reply_stream

        async def reply(query, **kwargs):
            runtime.files.write(summary, "# Current progress\n")
            async for chunk in original_reply(query, **kwargs):
                yield chunk

        monkeypatch.setattr(wrapper, "reply_stream", reply)
        sessions = []
        for number in range(2):
            event = Event(
                id=f"event-{number}",
                event_ts=runtime.now(),
                source="cron" if trigger_type == "cron" else "test",
                description="Next",
            )
            if trigger_type == "cron":
                await runtime._append(await runtime.definition("monitor"), [event])
            else:
                await runtime.append("monitor", event)
            response = await ProactiveAgentStep(app_context=context)(agent_id="monitor")
            assert response.success
            sessions.append(wrapper.turns[-1][1]["resume"])
            if number == 0:
                response = await AgentChatStep(app_context=context)(
                    RuntimeContext(stream_queue=asyncio.Queue()),
                    agent_id="monitor",
                    query="Explain this result",
                )
                assert response.success
                assert wrapper.turns[-1][1]["resume"] == sessions[0]

        assert sessions[0] != sessions[1]
        assert len(wrapper.created) == 2
        assert all(not {"resume", "fork_session"} & options.keys() for options in wrapper.created)
        assert (await runtime.sessions.current("monitor")).session_id == sessions[1]
        assert (await runtime.sessions.list("monitor"))["total"] == 2


@pytest.mark.asyncio
async def test_event_cooldown_accumulates_inputs_and_respects_batch_limit(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.update("monitor", {"min_trigger_interval_seconds": 10, "max_batch_events": 20})
        clock = [runtime.now()]
        monkeypatch.setattr(runtime, "now", lambda: clock[0])
        await runtime.append(
            "monitor",
            [
                {"id": f"event-{number}", "event_ts": runtime.now(), "source": "test", "description": "Next"}
                for number in range(2, 22)
            ],
        )
        batches = []
        completed = asyncio.Event()

        async def execute(_agent, events):
            batches.append([event["id"] for event in events])
            return ProactiveRunResult(notifications=[]), f"session-{len(batches)}"

        async def run_job(_name, **kwargs):
            answer = await runtime.run(kwargs["agent_id"], execute)
            completed.set()
            return Response(success=True, answer=answer)

        await runtime.tick(run_job)
        await asyncio.wait_for(completed.wait(), timeout=5)
        assert batches == [[f"event-{number}" for number in range(1, 21)]]
        snapshot = await runtime.inbox.snapshot("monitor")
        assert snapshot["pending_count"] == 1
        completed.clear()
        clock[0] = snapshot["receipt"].consume_ts + timedelta(seconds=9)
        await runtime.append(
            "monitor",
            {"id": "event-22", "event_ts": runtime.now(), "source": "test", "description": "During cooldown"},
        )
        await runtime.tick(run_job)
        assert "monitor" not in runtime.workers
        assert not completed.is_set()
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 2
        clock[0] += timedelta(seconds=1)
        await runtime.tick(run_job)
        await asyncio.wait_for(completed.wait(), timeout=5)
        assert batches[-1] == ["event-21", "event-22"]
        assert (await runtime.inbox.snapshot("monitor"))["pending_count"] == 0


@pytest.mark.asyncio
async def test_chat_read_updates_all_sources_idempotently_and_survives_restart(tmp_path):
    message_id = "reme-proactive-reminder-combined"
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.create({"name": "other", "description": "Other", "trigger_type": "event"})
        await runtime.append(
            "other",
            {"id": "other-1", "event_ts": runtime.now(), "source": "test", "description": "New"},
        )
        await runtime.run("other", AsyncMock(return_value=completed_result(runtime, "other")))
        await runtime.outbox.sync()
        refs = list(runtime.outbox.committed)
        for ref in refs:
            await runtime.outbox.update(ref, chat_message_ids=[message_id])
        result = await runtime.outbox.read("chat", message_id, mark_read=True)
        assert len(result["items"]) == 2
        first = {item["id"]: item["user_read_at"] for item in result["items"]}
        assert all(first.values())
        assert all(item["read_at"] is None for item in result["items"])
        assert len(runtime.notifications.pending) == 2
        again = await runtime.outbox.read("chat", message_id, mark_read=True)
        assert {item["id"]: item["user_read_at"] for item in again["items"]} == first
        with pytest.raises(ValueError):
            await runtime.outbox.read("monitor", message_id, mark_read=True)
    async with running_runtime(tmp_path) as (runtime, _, _):
        result = await runtime.outbox.read("chat", message_id)
        assert {item["id"]: item["user_read_at"] for item in result["items"]} == first
        assert all(
            frontmatter.loads(runtime.files.read(runtime.files.path(ref.file)))["chat_message_ids"] == [message_id]
            for ref in refs
        )


@pytest.mark.asyncio
async def test_read_acknowledgements_do_not_require_retained_reminder_sources(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        message_id = "reme-proactive-expired"
        assert await runtime.outbox.read("chat", message_id, mark_read=True) == {"id": message_id, "items": []}
        with pytest.raises(ValueError, match="no retained source"):
            await runtime.outbox.read("chat", message_id)


@pytest.mark.parametrize("trigger_type", ["event", "cron"])
@pytest.mark.asyncio
async def test_proactive_sessions_inherit_native_compaction_without_changing_network_policy(tmp_path, trigger_type):
    async with running_runtime(tmp_path) as (runtime, context, _):
        values = {"name": "monitor", "description": "Monitor", "trigger_type": trigger_type}
        if trigger_type == "cron":
            values["cron"] = "0 * * * *"
        await runtime.create(values)
        options = ProactiveAgentStep(app_context=context).session_options(await runtime.definition("monitor"))
        assert "model_auto_compact_token_limit" not in options["config"]
        online = trigger_type == "cron"
        assert options["config"]["sandbox_workspace_write"]["network_access"] is online
        assert options["config"]["web_search"] == ("live" if online else "disabled")


@pytest.mark.asyncio
async def test_sent_chat_message_keeps_source_links_before_consumption(tmp_path):
    from reme.schema import SessionMessage

    async with running_runtime(tmp_path) as (runtime, _, wrapper):
        await create_agent(runtime)
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        await runtime.outbox.sync()
        session_id = await runtime.sessions.ensure("chat", wrapper, {}, runtime.now())
        ids = list(runtime.notifications.pending)
        async with runtime.notifications.turn(session_id, "P0", ids) as turn:
            await runtime.notifications.read_turn(turn)
            message_id = f"reme-proactive-{turn.message_id}"

            async def append(*_args, **_kwargs):
                acknowledged = await runtime.outbox.read("chat", message_id, mark_read=True)
                assert acknowledged["items"] and all(item["user_read_at"] for item in acknowledged["items"])
                return SessionMessage(
                    id=message_id,
                    role="assistant",
                    content="Combined result",
                    source="proactive",
                )

            wrapper.append_session = AsyncMock(side_effect=append)
            await runtime.notifications.send(turn, "Combined result")
        assert not runtime.notifications.pending
        result = await runtime.outbox.read("chat", message_id, mark_read=True)
        assert result["items"][0]["user_read_at"]
        assert result["items"][0]["chat_message_ids"] == [message_id]


@pytest.mark.asyncio
async def test_committed_batch_is_not_retried_after_acknowledgement_error(tmp_path, monkeypatch):
    async with running_runtime(tmp_path) as (runtime, _, _):
        await create_agent(runtime)
        original = runtime.inbox.ack

        async def committed_then_interrupted(*args, **kwargs):
            await original(*args, **kwargs)
            raise RuntimeError("connection lost after durable acknowledgement")

        monkeypatch.setattr(runtime.inbox, "ack", committed_then_interrupted)
        with pytest.raises(RuntimeError, match="durable acknowledgement"):
            await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        monkeypatch.setattr(runtime.inbox, "ack", original)
        await runtime.append(
            "monitor",
            {"id": "next", "event_ts": runtime.now(), "source": "test", "description": "Next"},
        )
        execute = AsyncMock(return_value=completed_result(runtime))
        await runtime.run("monitor", execute)
        assert [item["id"] for item in execute.call_args.args[1]] == ["next"]
