"""Read-only event pages, scoped mailboxes and independent user receipts."""

# pylint: disable=missing-function-docstring,protected-access

from unittest.mock import AsyncMock

import pytest

from reme.core.context import RuntimeContext
from reme.schema.agent_view import AgentInboxRequest, ListTaskRemindersRequest, AgentReminderRequest
from reme.steps.proactive.agent import AgentInboxStep, ListTaskRemindersStep, AgentReminderStep
from .test_proactive_runtime import completed_result, create_agent, running_runtime
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
async def test_inbox_outbox_and_reminder_reads_share_canonical_files(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        await runtime.create({"agent_name": "other", "task_description": "Other"})
        now = runtime.now()
        hour = (now.hour + 1) % 24
        await runtime.update("monitor", {"notify_window": {"start": f"{hour:02}:00", "end": f"{hour:02}:01"}})
        await runtime.run("monitor", AsyncMock(return_value=completed_result(runtime)))
        task = ListTaskRemindersStep(app_context=context)
        inbox = AgentInboxStep(app_context=context)
        reminder = AgentReminderStep(app_context=context)
        outgoing = (await task(agent_name="monitor")).answer["items"]
        assert len(outgoing) == 1 and outgoing[0]["delivered_at"]
        identifier = outgoing[0]["id"]
        assert identifier == outgoing[0]["paths"][0]
        with pytest.raises(ValueError, match="not found"):
            await reminder(agent_name="other", message_id=identifier)
        (incoming,) = (await inbox()).answer["items"]
        assert incoming["reminder"] == outgoing[0]
        assert incoming["event"]["id"] == identifier
        assert (await reminder(message_id=identifier)).answer == outgoing[0]
        assert identifier in runtime.notifications.pending
        await runtime.notifications.consume([identifier])
        assert not (await inbox(state="pending")).answer["items"]
        assert (await inbox(state="processed")).answer["items"][0]["event"]["id"] == identifier
        assert (await task(agent_name="monitor", state="processed")).answer["items"][0]["read_at"]
        with pytest.raises(PermissionError):
            await task(RuntimeContext(_allowed_paths=["daily"]), agent_name="monitor")
        with pytest.raises(ValueError):
            await task(kind="instruction")


@pytest.mark.asyncio
async def test_inbox_endpoint_accepts_chat_and_tasks_and_preserves_processing_state(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        await create_agent(runtime)
        task = AgentInboxStep(app_context=context)
        assert (await task()).answer == {"items": [], "total": 0, "offset": 0, "limit": 20}
        assert (await task(agent_name="monitor")).answer["items"][0]["state"] == "pending"
        assert (await runtime.status("monitor"))["pending_events"] == 1
        batch = await runtime.inbox.claim("monitor")
        assert (await task(agent_name="monitor", state="processing")).answer["total"] == 1
        await runtime.inbox.nack("monitor", batch["token"])
        with pytest.raises(ValueError):
            await task(agent_name="missing")
        with pytest.raises(PermissionError):
            await task(RuntimeContext(_allowed_paths=["daily"]), agent_name="monitor")


@pytest.mark.parametrize(
    "schema,values",
    [
        (AgentInboxRequest, {"limit": 101}),
        (AgentInboxRequest, {"offset": True}),
        (ListTaskRemindersRequest, {}),
        (ListTaskRemindersRequest, {"agent_name": "monitor", "mark_read": True}),
        (AgentReminderRequest, {}),
        (AgentReminderRequest, {"message_id": "id", "state": "all"}),
        (AgentReminderRequest, {"message_id": "id", "mark_read": True}),
    ],
)
def test_resource_schemas_reject_ambiguous_or_retired_arguments(schema, values):
    with pytest.raises(ValueError):
        schema.model_validate(values)
