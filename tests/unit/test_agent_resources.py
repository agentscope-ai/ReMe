"""Read-only event pages, scoped mailboxes and independent user receipts."""

# pylint: disable=missing-function-docstring,protected-access

from unittest.mock import AsyncMock

import pytest

from reme.components import RuntimeContext
from reme.schema.agent_view import AgentEventListRequest, AgentMailboxRequest
from reme.steps.proactive.agent import AgentEventListStep, AgentMailboxStep
from .test_agent_runtime import completed_result, create_agent, running_runtime
from .test_agent_inbox import event, make_inbox


@pytest.mark.asyncio
async def test_event_pages_are_ordered_isolated_snapshots_without_claiming_or_writing(tmp_path):
    async with make_inbox(tmp_path) as inbox:
        await inbox.create("news")
        await inbox.append(
            "news",
            [event("early", event_ts="2026-11-11T20:30:00+08:00"), event("late", event_ts="2026-11-11T13:30:00+00:00")],
        )
        before = {path: path.read_bytes() for path in tmp_path.rglob("*.jsonl")}
        page = await inbox.read("news", limit=1)
        assert page["total"] == 2 and page["items"][0]["item"]["id"] == "late"
        page["items"][0]["item"]["paths"].clear()
        assert (await inbox.read("news"))["items"][0]["item"]["paths"]
        assert (await inbox.snapshot("news"))["in_flight_count"] == 0
        assert before == {path: path.read_bytes() for path in tmp_path.rglob("*.jsonl")}
        batch = await inbox.claim("news", 1)
        assert (await inbox.read("news", state="processing"))["items"][0]["item"]["id"] == "early"
        assert (await inbox.read("news", state="pending"))["items"][0]["item"]["id"] == "late"
        await inbox.ack("news", batch["token"])
        processed = (await inbox.read("news", state="processed"))["items"]
        assert processed[0]["consume_ts"] and processed[0]["item"]["id"] == "early"
        assert not (await inbox.read("news", offset=2))["items"]


@pytest.mark.asyncio
async def test_mailbox_views_share_canonical_mail_and_queries_never_publish_or_consume(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        await runtime.create({"name": "other", "trigger_type": "event", "description": "Other"})
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        task = AgentMailboxStep(app_context=context)
        outgoing = (await task(agent_id="monitor")).answer["items"]
        assert len(outgoing) == 1 and outgoing[0]["delivered_at"] is None
        assert not (await task()).answer["items"]
        assert not runtime.notifications.pending
        with pytest.raises(ValueError, match="not found"):
            await task(action="read", agent_id="other", message_id=outgoing[0]["id"], mark_read=True)
        await runtime.outbox.sync()
        incoming = (await task()).answer["items"]
        assert incoming == (await task(agent_id="monitor")).answer["items"]
        identifier = incoming[0]["id"]
        read = (await task(action="read", message_id=identifier, mark_read=True)).answer
        assert read["user_read_at"] and read["read_at"] is None and identifier in runtime.notifications.pending
        await runtime.notifications.consume([identifier])
        assert not (await task(state="pending")).answer["items"]
        assert (await task(state="processed")).answer["items"][0]["id"] == identifier
        assert (await task(agent_id="monitor", state="processed")).answer["items"][0]["user_read_at"]
        with pytest.raises(PermissionError):
            await task(RuntimeContext(_allowed_paths=["daily"]))
        with pytest.raises(ValueError):
            await task(kind="instruction")


@pytest.mark.asyncio
async def test_event_endpoint_preserves_inputs_and_rejects_chat_unknown_agents_and_permissions(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        task = AgentEventListStep(app_context=context)
        assert (await task(agent_id="monitor")).answer["items"][0]["state"] == "pending"
        assert (await runtime.status("monitor"))["pending_events"] == 1
        for agent_id in ["chat", "missing"]:
            with pytest.raises(ValueError):
                await task(agent_id=agent_id)
        with pytest.raises(PermissionError):
            await task(RuntimeContext(_allowed_paths=["daily"]), agent_id="monitor")


@pytest.mark.parametrize(
    "schema,values",
    [
        (AgentEventListRequest, {"limit": 101}),
        (AgentEventListRequest, {"offset": True}),
        (AgentMailboxRequest, {"action": "read"}),
        (AgentMailboxRequest, {"action": "read", "message_id": "id", "state": "all"}),
        (AgentMailboxRequest, {"mark_read": True}),
        (AgentMailboxRequest, {"task_id": "monitor"}),
    ],
)
def test_resource_schemas_reject_ambiguous_or_retired_arguments(schema, values):
    with pytest.raises(ValueError):
        schema.model_validate(values)
