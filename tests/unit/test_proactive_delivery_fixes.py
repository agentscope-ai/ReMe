"""Regression coverage for bounded retries and durable chat read acknowledgements."""

# pylint: disable=missing-function-docstring,protected-access

from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.schema.proactive_agent import ProactiveAgentConfig
from reme.steps.proactive.agent import ProactiveAgentStep
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


@pytest.mark.parametrize("size", [0, -1, True, "2", 1.5])
def test_event_batch_size_rejects_invalid_values(size):
    with pytest.raises(ValueError):
        ProactiveAgentConfig(name="monitor", trigger_type="event", max_batch_events=size)


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
