"""Mailbox views preserve receiver-owned state and canonical record content."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import pytest

from reme.core.context import RuntimeContext
from reme.schema.agent_view import AgentInboxRequest, ListTaskRemindersRequest, AgentReminderRequest
from reme.steps.proactive.api import AgentInboxStep, ListTaskRemindersStep, AgentReminderStep
from reme.steps.proactive.chat import ProcessChatNotificationsStep
from .proactive_helpers import running_system, emit_notification, queue_event


@pytest.mark.asyncio
async def test_mailbox_reads_do_not_consume_or_rewrite_and_reflect_source_edits(tmp_path):
    async with running_system(tmp_path) as (manager, context, _):
        ref = await emit_notification(manager, context)
        chat = context.components["agent_runtime"]["Remy"]
        inbox = AgentInboxStep(app_context=context, agent_runtime="Remy")
        outbox = ListTaskRemindersStep(app_context=context, agent_runtime="Remy")
        detail = AgentReminderStep(app_context=context, agent_runtime="Remy")
        before = {path: path.read_bytes() for path in tmp_path.rglob("*.jsonl")}
        rows = (await outbox(agent_name="monitor")).answer["items"]
        assert rows == (await inbox()).answer["items"] and rows[0]["state"] == "pending"
        assert (await detail(message_id=ref.record_path)).answer["message"] == "Verified result"
        assert before == {path: path.read_bytes() for path in tmp_path.rglob("*.jsonl")}
        notifications = ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")
        async with chat.notification_turn(
            await chat.ensure_session({}),
            "P1",
            [ref.id],
            format_item=notifications.format_item,
        ):
            assert (await inbox(state="processing")).answer["total"] == 1
        path = tmp_path / ref.record_path
        path.write_text(path.read_text().replace("Verified result", "Edited"))
        assert (await detail(message_id=ref.record_path)).answer["message"] == "Edited"
        batch = await chat.inbox.claim()
        await chat.inbox.ack(batch["token"], metadata={"decision": "skip"})
        assert (await outbox(agent_name="monitor", state="processed")).answer["items"][0]["read_at"]
        assert not (await inbox(state="pending")).answer["items"]
        with pytest.raises(PermissionError):
            await inbox(RuntimeContext(_allowed_paths=["daily"]))


@pytest.mark.asyncio
async def test_task_queue_view_preserves_event_metadata_and_claims(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_type": "event", "agent_name": "monitor"}]) as (
        manager,
        context,
        _,
    ):
        await queue_event(manager)
        runtime = await manager.resolve("monitor")
        step = AgentInboxStep(app_context=context, agent_runtime="Remy")
        page = (await step(agent_name="monitor")).answer
        assert page["items"][0]["event"]["id"] == "event-1" and page["items"][0]["state"] == "pending"
        batch = await runtime.inbox.claim()
        assert (await step(agent_name="monitor", state="processing")).answer["total"] == 1
        assert (await runtime.inbox.snapshot())["in_flight_count"] == 1
        await runtime.inbox.nack(batch["token"])
        with pytest.raises(ValueError):
            await step(agent_name="missing")


@pytest.mark.parametrize(
    "schema,values",
    [
        (AgentInboxRequest, {"limit": 101}),
        (AgentInboxRequest, {"offset": True}),
        (ListTaskRemindersRequest, {}),
        (ListTaskRemindersRequest, {"agent_name": "monitor", "mark_read": True}),
        (AgentReminderRequest, {}),
        (AgentReminderRequest, {"message_id": "id", "state": "all"}),
    ],
)
def test_resource_schemas_reject_ambiguous_arguments(schema, values):
    with pytest.raises(ValueError):
        schema.model_validate(values)
