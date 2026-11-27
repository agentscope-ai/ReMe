"""Mailbox views preserve receiver-owned state and canonical record content."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from unittest.mock import AsyncMock

import pytest

from reme.constants import PROACTIVE_MESSAGE_ID_PREFIX
from reme.core.context import RuntimeContext
from reme.schema.agent_view import AgentInboxRequest, ListTaskRemindersRequest, AgentReminderRequest
from reme.steps.proactive.agent_api import AgentInboxStep, ListTaskRemindersStep, AgentReminderStep
from reme.steps.proactive.chat import ProcessChatNotificationsStep
from .proactive_helpers import running_system, emit_notification, queue_event


@pytest.mark.parametrize("agent_name", [None, "monitor"])
@pytest.mark.asyncio
async def test_reminder_pages_and_combined_detail_share_one_chat_outbox_read(tmp_path, monkeypatch, agent_name):
    async with running_system(tmp_path) as (manager, context, _):
        references = [await emit_notification(manager, context, suffix=number) for number in range(3)]
        chat = context.components["agent_runtime"]["Remy"]
        notifications = ProcessChatNotificationsStep(app_context=context, agent_runtime="Remy")
        async with chat.notification_turn(
            await chat.ensure_session({}),
            "P1",
            [reference.id for reference in references[:2]],
            format_item=notifications.format_item,
        ) as turn:
            await chat.submit_reply(turn.id, "send", "Combined reminder")
        delivery = (await chat.outbox.entries())[0]
        message_id = PROACTIVE_MESSAGE_ID_PREFIX + delivery.message_id
        entries = AsyncMock(wraps=chat.outbox.entries)
        monkeypatch.setattr(chat.outbox, "entries", entries)
        step_class = ListTaskRemindersStep if agent_name else AgentInboxStep
        step = step_class(app_context=context, agent_runtime="Remy")
        values = {"agent_name": agent_name} if agent_name else {}
        page = (await step(**values, limit=1)).answer
        assert page["total"] == 3 and len(page["items"]) == 1
        assert entries.await_count == 1
        entries.reset_mock()
        full = (await step(**values)).answer
        assert page["items"] == full["items"][:1] and entries.await_count == 1
        for item in full["items"]:
            expected = [message_id] if item["id"] in {ref.record_path for ref in references[:2]} else []
            assert item["chat_message_ids"] == expected
        entries.reset_mock()
        detail = (await AgentReminderStep(app_context=context, agent_runtime="Remy")(message_id=message_id)).answer
        assert {item["id"] for item in detail["items"]} == {ref.record_path for ref in references[:2]}
        assert all(item["chat_message_ids"] == [message_id] for item in detail["items"])
        assert entries.await_count == 1


@pytest.mark.parametrize("missing_definition", [False, True])
@pytest.mark.asyncio
async def test_deleted_sender_reminders_remain_readable_without_starting_a_runtime(tmp_path, missing_definition):
    async with running_system(tmp_path) as (manager, context, wrapper):
        reference = await emit_notification(manager, context)
        runtime = await manager.resolve("monitor")
        await manager.delete("monitor")
        if missing_definition:
            runtime.config_path.unlink()
        before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        response = await AgentReminderStep(app_context=context, agent_runtime="Remy")(
            message_id=reference.record_path,
        )
        assert response.success and response.answer["message"] == "Verified result"
        assert not manager.runtimes and not wrapper.turns
        assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


@pytest.mark.asyncio
async def test_reminder_detail_rejects_a_delivery_saved_under_another_identity(tmp_path):
    async with running_system(tmp_path) as (manager, context, _):
        first = await emit_notification(manager, context)
        second = await emit_notification(manager, context, suffix=1)
        runtime = await manager.resolve("monitor")
        deliveries = {row.notification.id: row for row in await runtime.outbox.entries()}
        path = runtime.outbox.path(deliveries[first.id].id)
        contents = runtime.outbox.path(deliveries[second.id].id).read_bytes()
        path.write_bytes(contents)
        with pytest.raises(ValueError, match="identity"):
            await AgentReminderStep(app_context=context, agent_runtime="Remy")(message_id=first.record_path)
        assert path.read_bytes() == contents


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
