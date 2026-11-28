"""Whole-flow subscriptions use real queues and outboxes, with an isolated native backend."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
import json
import os
from datetime import timedelta
from hashlib import sha256
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from reme.components.job import BaseJob, StreamJob
from reme.components.proactive.connector import ConnectorManager
from reme.components.proactive.inbox import BaseInbox, EventInbox
from reme.components.proactive.overview_events import OverviewEvents
from reme.core.context import ApplicationContext, RuntimeContext
from reme.schema import Event
from reme.schema.application_config import JobConfig
from reme.schema.agent_outbox import ChatDelivery
from reme.schema.proactive_overview import OverviewSnapshot, OverviewUpdate, WatchProactiveOverviewRequest
from reme.schema.proactive_trace import SourceRef
from reme.steps.proactive.overview import ProactiveOverviewViews, WatchProactiveOverviewStep
from reme.steps.proactive.connector.base import BaseConnectorStep, update_connector_control
from reme.utils.proactive_record import RoundRecord
from .proactive_helpers import running_system, emit_notification, queue_event


def overview_views(manager, context):
    step = WatchProactiveOverviewStep(
        app_context=context,
        agent_manager=manager,
        agent_runtime=manager.notification_target,
        connector_manager=context.components.get("connector_manager", {}).get("default")
        or ConnectorManager(app_context=context, agent_manager=manager),
    )
    step.context = RuntimeContext()
    return ProactiveOverviewViews(step)


@pytest.mark.asyncio
async def test_snapshot_covers_all_sections_without_model_calls_or_consumption(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        wrapper,
    ):
        await queue_event(manager)
        views = overview_views(manager, context)
        events = views.events()
        try:
            snapshot = OverviewSnapshot.model_validate(await anext(events))
            value = snapshot.overview
            assert {item.agent_name for item in value.agents} == {"monitor", "Remy"}
            assert {
                "sources",
                "agents",
                "event_inbox",
                "remy_inbox",
                "reminders",
                "workspace",
            } <= value.model_fields_set
            assert value.event_inbox.waiting_count == 1 and value.event_inbox.processing_count == 0
            assert value.event_inbox.items[0].event.source == "test"
            assert value.event_inbox.items[0].received_at
            assert not value.event_inbox.items[0].run_ids
            assert any(item.role == "task" and item.agent_name == "monitor" for item in value.workspace)
            assert not wrapper.created and not wrapper.turns
            assert await manager.runtimes["monitor"].inbox.pending_ids() == {"event-1"}
            assert not await manager.runtimes["monitor"].inbox.receipts()
        finally:
            await events.aclose()
        assert not manager.overview_events.listeners
        assert not manager.notification_target.live_turns.listeners


@pytest.mark.asyncio
async def test_subscribers_share_idle_projections_and_owner_changes_invalidate_them(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        _,
    ):
        views = overview_views(manager, context)
        read = AsyncMock(wraps=views.read)
        monkeypatch.setattr(views, "read", read)
        first = views.events(interval=0.01, heartbeat=0.05)
        second = views.events(interval=0.01, heartbeat=0.05)
        try:
            await anext(first)
            await anext(second)
            count = read.await_count
            assert (await anext(first))["type"] == "heartbeat"
            assert read.await_count == count
            await queue_event(manager)
            update = await anext(first)
            assert update["type"] == "update" and update["transitions"][0]["type"] == "event_admitted"
            count = read.await_count
            assert (await anext(second))["type"] == "update"
            assert read.await_count == count
        finally:
            await first.aclose()
            await second.aclose()


@pytest.mark.asyncio
async def test_epoch_reset_during_refresh_discards_the_previous_projection(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        _,
    ):
        views = overview_views(manager, context)
        events = views.events(interval=0.01)
        try:
            initial = await anext(events)
            task = manager.runtimes["monitor"].task_path
            read = views.read
            reset = False

            async def reset_during_read(names=None):
                nonlocal reset
                value = await read(names)
                if not reset:
                    reset = True
                    manager.overview_events.reset()
                    stamp = task.stat().st_mtime + 10
                    os.utime(task, (stamp, stamp))
                return value

            manager.overview_events.projections.clear()
            monkeypatch.setattr(views, "read", reset_during_read)
            refreshed = await anext(events)
            assert refreshed["type"] == "snapshot" and refreshed["epoch"] != initial["epoch"]
            before = next(item["mtime"] for item in initial["overview"]["workspace"] if item["role"] == "task")
            after = next(item["mtime"] for item in refreshed["overview"]["workspace"] if item["role"] == "task")
            assert before != after
        finally:
            await events.aclose()


@pytest.mark.asyncio
async def test_waiting_status_matches_scheduler_eligibility_and_keeps_retry_separate(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        _,
    ):
        runtime = await manager.resolve("monitor")
        views = overview_views(manager, context)

        async def status():
            return next(item for item in (await views.read())["agents"] if item["agent_name"] == "monitor")

        assert (await status())["waiting_reason"] == "waiting_input"
        await manager.update("monitor", {"enabled": False})
        assert (await status())["waiting_reason"] == "paused"
        await manager.update("monitor", {"enabled": True})
        await queue_event(manager)
        runtime.failed_round("temporary failure")
        value = await status()
        assert value["waiting_reason"] == "retry_backoff" and value["retry_at"] == value["waiting_until"]
        assert value["next_trigger_at"] is None
        runtime.completed_round()
        assert (await status())["waiting_reason"] is None
        batch = await runtime.inbox.claim()
        await runtime.inbox.ack(batch["token"])
        await manager.update("monitor", {"min_trigger_interval_seconds": 60})
        await queue_event(manager, identifier="second")
        value = await status()
        assert value["waiting_reason"] == "cooldown" and value["retry_at"] is None
        chat = manager.notification_target
        chat.retry_at = chat.now() + timedelta(seconds=5)
        remy = next(item for item in (await views.read())["agents"] if item["agent_name"] == "Remy")
        assert remy["retry_at"] == chat.retry_at.isoformat() and remy["next_trigger_at"] is None
        assert runtime.waiting_state(await runtime.definition(), await runtime.inbox.snapshot())[
            1
        ] > runtime.now() + timedelta(seconds=50)


@pytest.mark.asyncio
@pytest.mark.parametrize("preset", [False, True])
async def test_current_source_targets_do_not_include_historical_recipients(tmp_path, monkeypatch, preset):
    class Collector(BaseConnectorStep):
        """Expose a preset source without performing collection."""

        source = "overview_targets"

    async def no_network(_source):
        return None

    async with running_system(
        tmp_path,
        runtimes=[{"agent_name": name, "agent_type": "event"} for name in ("monitor", "other")],
    ) as (manager, context, _):
        connector = ConnectorManager(name="default", app_context=context, agent_manager=manager)
        monkeypatch.setattr(connector, "_run", no_network)
        context.components["connector_manager"] = {"default": connector}
        async with connector:
            views = overview_views(manager, context)
            if preset:
                context.registry.add("overview_targets_step", Collector, owner=__name__)
                context.app_config.jobs["overview_targets"] = JobConfig(
                    backend="base",
                    steps=[
                        {
                            "backend": "overview_targets_step",
                            "connector_job": "overview_targets",
                            "runtime_names": ["monitor"],
                        },
                    ],
                )
                source_id = "preset:overview_targets"
            else:
                source = await connector.create(
                    {
                        "name": "Original name",
                        "runtime_names": ["monitor"],
                        "binding": {"type": "sse", "url": "https://example.invalid/events"},
                    },
                )
                source_id = source["source_id"]
            await manager.append_events(
                "monitor",
                [Event(id="historical", event_ts=manager.now(), source="producer", description="Observed")],
                source_ref=SourceRef(source_id=source_id, name="Original name", trigger_type="event"),
            )
            for targets in (["other"], []):
                if preset:
                    await update_connector_control(views.step, "overview_targets", runtime_names=targets)
                else:
                    await connector.update(source_id, runtime_names=targets)
                overview = await views.read()
                source = next(item for item in overview["sources"] if item["source_id"] == source_id)
                assert source["runtime_names"] == targets
                assert source["last_admitted_at"]
                historical = overview["event_inbox"]["items"][0]
                assert historical["source_refs"][0]["name"] == "Original name"
                filtered = await views.read(["monitor"])
                assert not any(item["source_id"] == source_id for item in filtered["sources"])


@pytest.mark.asyncio
async def test_workspace_keeps_recent_records_and_current_provenance_instead_of_all_history(tmp_path):
    async with running_system(tmp_path) as (manager, context, _):
        references = [await emit_notification(manager, context, suffix=index) for index in range(8)]
        views = overview_views(manager, context)
        events = views.events()
        try:
            snapshot = (await anext(events))["overview"]
            records = {item["path"] for item in snapshot["workspace"] if item["agent_name"] == "monitor"}
            assert {reference.record_path for reference in references[-3:]} <= records
            assert not {reference.record_path for reference in references[:-3]} & records
            assert len(await manager.runtimes["monitor"].outbox.entries()) == 8
            await emit_notification(manager, context, suffix=8)
            update = await anext(events)
            removed = {
                item["id"] for item in update["changes"] if item["collection"] == "workspace" and item["value"] is None
            }
            assert f"monitor:record:{references[-3].record_path}" in removed
            runtime = manager.runtimes["monitor"]
            runtime.last_run = {
                "run_id": "retained-run",
                "phase": "completed",
                "record_paths": [references[0].record_path],
            }
            retained = await views.read()
            assert any(item["path"] == references[0].record_path for item in retained["workspace"])
            chat = manager.notification_target
            session_id = await chat.ensure_session(chat.session_options())
            reference = references[1]
            key = "\0".join((session_id, reference.id))
            await chat.outbox.prepare(
                ChatDelivery(
                    id="old-record-message",
                    notification_ids=[reference.id],
                    record_versions={reference.record_path: "v1"},
                    session_id=session_id,
                    chat_revision=0,
                    content="Recent message about an older record",
                    message_id="reminder-" + sha256(key.encode()).hexdigest(),
                    created_at=manager.now(),
                ),
            )
            await chat.outbox.deliver("old-record-message")
            recent = await views.read()
            assert recent["reminders"][0]["run_ids"] == [reference.trace.run_id]
            assert any(item["path"] == reference.record_path for item in recent["workspace"])
        finally:
            await events.aclose()


@pytest.mark.asyncio
async def test_durable_admission_time_survives_retry_receipt_and_restart(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    event = Event(id="one", event_ts="2000-01-01T00:00:00Z", source="producer", description="old observation")
    ref = SourceRef(source_id="custom", name="Original name", trigger_type="event")
    inbox = EventInbox(directory="inbox", agent_name="monitor", app_context=context)
    async with inbox:
        await inbox.append(event, source_ref=ref)
        initial = (await inbox.overview())["items"][0]
        assert initial["received_at"] != event.event_ts.isoformat()
        await inbox.append(event, source_ref=ref.model_copy(update={"name": "Renamed"}))
        assert (await inbox.overview())["items"][0] == initial
        batch = await inbox.claim()
        assert batch["source_refs"]["one"]["name"] == "Original name"
        counts = await inbox.overview()
        assert counts["waiting_count"] == 0 and counts["processing_count"] == 1
        assert (await inbox.snapshot())["pending_count"] == 1
        await inbox.ack(batch["token"], metadata={"run_id": batch["token"]})
    async with EventInbox(directory="inbox", agent_name="monitor", app_context=context) as restarted:
        row = (await restarted.overview())["items"][0]
        assert row["received_at"] == initial["received_at"]
        assert row["source_ref"]["name"] == "Original name"
        assert row["run_id"] == batch["token"] and row["state"] == "processed"
        assert (await restarted.read())["items"][0]["item"] == event.model_dump(mode="json")


@pytest.mark.asyncio
async def test_generic_queue_payloads_are_not_interpreted_as_proactive_events(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    item = {"id": "generic", "source": {"nested": True}, "trace": True}
    async with BaseInbox(directory="inbox", agent_name="generic", app_context=context) as inbox:
        await inbox.append(item)
        batch = await inbox.claim()
        assert set(batch) == {"token", "items"}
        await inbox.ack(batch["token"])
        assert (await inbox.read())["items"][0]["item"] == item


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_envelope", ["unwrapped", "missing_source", "missing_timezone"])
async def test_invalid_admission_envelopes_fail_closed(tmp_path, invalid_envelope):
    directory = tmp_path / "inbox"
    directory.mkdir()
    event = Event(event_ts="2000-01-01T00:00:00Z", source="sample", description="observation")
    value = event.model_dump(mode="json")
    if invalid_envelope != "unwrapped":
        value = {
            "item": value,
            "received_at": "2026-11-28T08:00:00" if invalid_envelope == "missing_timezone" else "2026-11-28T08:00:00Z",
            "source_ref": None,
        }
    (directory / "inbox_unread.jsonl").write_text(json.dumps(value) + "\n")
    context = ApplicationContext(workspace_dir=str(tmp_path))
    with pytest.raises(ValueError):
        async with EventInbox(directory="inbox", app_context=context):
            pass


@pytest.mark.asyncio
async def test_short_lived_actions_are_retained_without_replaying_on_reconnect(tmp_path):
    async with running_system(tmp_path) as (manager, context, wrapper):
        views = overview_views(manager, context)
        events = views.events()
        try:
            first = await anext(events)
            reference = await emit_notification(manager, context)
            update = OverviewUpdate.model_validate(await asyncio.wait_for(anext(events), 2))
            assert update.previous_revision == first["revision"]
            kinds = {value.type for value in update.transitions}
            assert {
                "event_admitted",
                "input_claimed",
                "input_processed",
                "record_committed",
                "notification_prepared",
                "notification_delivered",
            } <= kinds
            assert len({value.id for value in update.transitions}) == len(update.transitions)
            claimed = next(value for value in update.transitions if value.type == "input_claimed")
            assert claimed.run_id == reference.trace.run_id
            assert next(value for value in update.transitions if value.type == "record_committed").paths == [
                reference.record_path,
            ]
            snapshot = await views.read()
            assert snapshot["remy_inbox"]["waiting_count"] == 1
            assert snapshot["remy_inbox"]["items"][0]["run_ids"] == [claimed.run_id]
            assert not snapshot["reminders"]
            agent = next(value for value in snapshot["agents"] if value["agent_name"] == "monitor")
            assert agent["last_run"]["record_paths"] == [reference.record_path]
            assert agent["delivered_count"] == 1
            assert not wrapper.created and not wrapper.turns
            resumed = views.events()
            try:
                restored = await anext(resumed)
                assert restored["type"] == "snapshot" and "transitions" not in restored
                assert restored["overview"]["remy_inbox"]["waiting_count"] == 1
            finally:
                await resumed.aclose()
        finally:
            await events.aclose()


@pytest.mark.asyncio
async def test_snapshot_boundary_retains_concurrent_admission(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        _,
    ):
        views = overview_views(manager, context)
        original = views.read
        injected = False

        async def read(names=None):
            nonlocal injected
            value = await original(names)
            if not injected:
                injected = True
                await queue_event(manager)
            return value

        monkeypatch.setattr(views, "read", read)
        events = views.events()
        try:
            snapshot = await anext(events)
            assert snapshot["revision"] < manager.overview_events.revision
            assert snapshot["overview"]["event_inbox"]["waiting_count"] == 0
            concurrent = await anext(events)
            assert concurrent["previous_revision"] == snapshot["revision"]
            assert [value["event_ids"] for value in concurrent["transitions"]] == [["event-1"]]
            await queue_event(manager, identifier="event-2")
            update = await anext(events)
            assert update["previous_revision"] == concurrent["revision"]
            assert [value["event_ids"] for value in update["transitions"]] == [["event-2"]]
        finally:
            await events.aclose()


@pytest.mark.asyncio
async def test_subscription_progresses_while_every_snapshot_races_with_new_input(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        _,
    ):
        views = overview_views(manager, context)
        original = views.read
        count = 0

        async def busy_read(names=None):
            nonlocal count
            value = await original(names)
            count += 1
            await queue_event(manager, identifier=f"event-{count}")
            return value

        monkeypatch.setattr(views, "read", busy_read)
        events = views.events()
        try:
            first = await asyncio.wait_for(anext(events), 2)
            assert first["type"] == "snapshot"
            update = await asyncio.wait_for(anext(events), 2)
            assert update["type"] == "update"
            assert update["previous_revision"] == first["revision"]
            assert [value["event_ids"] for value in update["transitions"]] == [["event-1"]]
        finally:
            await events.aclose()


@pytest.mark.asyncio
async def test_slow_reader_resnapshots_and_duplicate_delivery_has_no_new_action(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        _,
    ):
        views = overview_views(manager, context)
        events = views.events()
        try:
            await anext(events)
            await queue_event(manager)
            value = manager.overview_events.revision
            item = (await manager.runtimes["monitor"].inbox.read())["items"][0]["item"]
            await manager.append_events("monitor", [item])
            assert manager.overview_events.revision == value
            for _ in range(manager.overview_events.capacity + 1):
                manager.overview_events.publish()
            event = await anext(events)
            assert event["type"] == "snapshot" and event["overview"]["event_inbox"]["waiting_count"] == 1
            assert "transitions" not in event
        finally:
            await events.aclose()


@pytest.mark.asyncio
async def test_filter_and_same_event_sent_to_multiple_receivers(tmp_path):
    configs = [{"agent_name": name, "agent_type": "event"} for name in ("one", "two")]
    async with running_system(tmp_path, runtimes=configs) as (manager, context, _):
        event = Event(id="same", event_ts=manager.now(), source="producer", description="fanout")
        await manager.deliver_events(["one", "two"], [event])
        views = overview_views(manager, context)
        snapshot = await views.read()
        assert {(row["agent_name"], row["id"]) for row in snapshot["event_inbox"]["items"]} == {
            ("one", "same"),
            ("two", "same"),
        }
        assert snapshot["event_inbox"]["waiting_count"] == 2
        filtered = await views.read(["one"])
        assert {row["agent_name"] for row in filtered["agents"]} == {"one", "Remy"}
        assert filtered["event_inbox"]["waiting_count"] == 1
        with pytest.raises(ValueError, match="Unknown"):
            await views.read(["missing"])


@pytest.mark.asyncio
async def test_time_source_and_retry_attempts_have_distinct_identities(tmp_path):
    configs = [{"agent_name": "timer", "agent_type": "time", "schedule": "0 9 * * *"}]
    async with running_system(tmp_path, runtimes=configs) as (manager, context, _):
        runtime = manager.runtimes["timer"]
        await runtime.inbox.append(Event(id="scheduled", event_ts=manager.now(), source="time", description="due"))
        first = await runtime.inbox.claim()
        runtime.last_started_at = runtime.now()
        runtime.begin_run(first)
        runtime.finish_run("failed", "temporary failure")
        await runtime.inbox.nack(first["token"])
        second = await runtime.inbox.claim()
        runtime.begin_run(second)
        assert first["token"] != second["token"]
        snapshot = await overview_views(manager, context).read()
        agent = next(value for value in snapshot["agents"] if value["agent_name"] == "timer")
        assert agent["last_run"]["run_id"] == first["token"] and agent["last_run"]["error"] == "temporary failure"
        assert agent["active_run"]["run_id"] == second["token"]
        assert snapshot["event_inbox"]["waiting_count"] == 0 and snapshot["event_inbox"]["processing_count"] == 1
        source = next(value for value in snapshot["sources"] if value["source_id"] == "time:timer")
        assert source["trigger_type"] == "time" and source["schedule"] == "0 9 * * *"


@pytest.mark.asyncio
@pytest.mark.parametrize("recovered", [False, True])
async def test_only_sent_messages_appear_and_merged_message_keeps_all_inputs(tmp_path, recovered):
    async with running_system(tmp_path) as (manager, context, _):
        refs = [await emit_notification(manager, context, name=name) for name in ("one", "two")]
        chat = manager.notification_target
        session_id = await chat.ensure_session(chat.session_options())
        ids = [ref.id for ref in refs]
        key = "\0".join((session_id, *sorted(ids)))
        delivery = ChatDelivery(
            id="merged",
            notification_ids=ids,
            record_versions={ref.record_path: "v1" for ref in refs},
            session_id=session_id,
            chat_revision=0,
            content="Combined update",
            message_id="reminder-" + sha256(key.encode()).hexdigest(),
            created_at=manager.now(),
        )
        await chat.outbox.prepare(delivery)
        views = overview_views(manager, context)
        assert not (await views.read())["reminders"]
        if recovered:
            await chat.agent_wrapper.append_session(session_id, delivery.content, message_id=delivery.message_id)
        before = manager.overview_events.revision
        await chat.outbox.deliver(delivery.id)
        actions = [value for value in manager.overview_events.since(before) if value["type"] == "reminder_sent"]
        assert len(actions) == (0 if recovered else 1)
        snapshot = await views.read()
        assert len(snapshot["reminders"]) == 1
        reminder = snapshot["reminders"][0]
        assert reminder["summary"] == "Combined update" and set(reminder["notification_ids"]) == set(ids)
        assert set(reminder["run_ids"]) == {ref.trace.run_id for ref in refs}
        assert reminder["status"] == "sent" and "read_at" not in reminder
        assert all(row["state"] == "pending" and row["chat_message_ids"] for row in snapshot["remy_inbox"]["items"])


@pytest.mark.asyncio
async def test_delivery_failure_keeps_record_and_reports_error_until_transport_recovers(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, context, _):
        chat = manager.notification_target
        append = chat.inbox.append

        async def fail(*_args, **_kwargs):
            raise TimeoutError("receiver full")

        monkeypatch.setattr(chat.inbox, "append", fail)
        reference = await emit_notification(manager, context)
        views = overview_views(manager, context)
        snapshot = await views.read()
        agent = next(value for value in snapshot["agents"] if value["agent_name"] == "monitor")
        assert agent["pending_delivery_count"] == 1
        assert agent["deliveries"][0]["last_error"] == "receiver full"
        assert snapshot["remy_inbox"]["waiting_count"] == 0
        assert any(row["path"] == reference.record_path for row in snapshot["workspace"])
        assert not any(value["type"] == "notification_delivered" for value in manager.overview_events.since(0))
        monkeypatch.setattr(chat.inbox, "append", append)
        await manager.runtimes["monitor"].recover_handoffs()
        assert (await views.read())["remy_inbox"]["waiting_count"] == 1


@pytest.mark.asyncio
async def test_receiver_admission_is_observed_when_sender_confirmation_fails(tmp_path, monkeypatch):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        _,
    ):
        runtime = manager.runtimes["monitor"]
        save = runtime.outbox.save

        async def fail_confirmation(delivery):
            if delivery.status == "delivered":
                raise OSError("confirmation unavailable")
            return await save(delivery)

        monkeypatch.setattr(runtime.outbox, "save", fail_confirmation)
        reference = await emit_notification(manager, context)
        actions = [value for value in manager.overview_events.since(0) if value["type"] == "notification_delivered"]
        assert len(actions) == 1
        value = await overview_views(manager, context).read()
        row = value["remy_inbox"]["items"][0]
        assert row["id"] == reference.id and row["delivery"]["status"] == "prepared"
        assert actions[0]["occurred_at"] == row["received_at"]
        monkeypatch.setattr(runtime.outbox, "save", save)
        await runtime.recover_handoffs()
        actions = [value for value in manager.overview_events.since(0) if value["type"] == "notification_delivered"]
        assert len(actions) == 1


@pytest.mark.asyncio
async def test_record_without_reminder_and_restart_preserve_execution_provenance(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        _,
    ):
        await queue_event(manager)
        runtime = manager.runtimes["monitor"]
        batch = await runtime.inbox.claim()
        runtime.last_started_at = runtime.now()
        record = RoundRecord(runtime.files)
        path = runtime.daily_directory / runtime.now().date().isoformat() / "monitor_000001.md"
        await record.prepare(path)
        runtime.files.write(path, "---\nreminder_priority: null\n---\nA useful record, without a notification.\n")
        record.capture()
        await runtime.commit_round(batch, None, "saved-session", record)
        relative = runtime.to_workspace_relative(path)
        assert relative in await runtime.receipt_record_paths()
        value = await overview_views(manager, context).read()
        assert not value["reminders"] and value["remy_inbox"]["waiting_count"] == 0
        old_epoch = manager.overview_events.epoch
    async with running_system(tmp_path) as (manager, context, wrapper):
        assert manager.overview_events.epoch != old_epoch
        value = await overview_views(manager, context).read()
        agent = next(item for item in value["agents"] if item["agent_name"] == "monitor")
        assert agent["last_run"]["record_paths"] == [relative]
        assert agent["last_run"]["event_ids"] == ["event-1"]
        assert agent["last_run"]["run_id"] == batch["token"]
        assert agent["last_run"]["notification_requested"] is False
        assert not wrapper.created


@pytest.mark.asyncio
async def test_public_stream_job_emits_snapshot_updates_done_and_honors_permissions(tmp_path):
    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        wrapper,
    ):
        context.components.setdefault(
            "connector_manager",
            {"default": ConnectorManager(app_context=context, agent_manager=manager)},
        )
        config = context.app_config.jobs["watch_proactive_overview"].model_dump(exclude={"backend"})
        job = StreamJob(name="watch_proactive_overview", app_context=context, **config)
        queue = asyncio.Queue(maxsize=1)
        async with job:
            task = asyncio.create_task(job(stream_queue=queue))
            try:
                first = await asyncio.wait_for(queue.get(), 2)
                assert first.chunk_type == "data" and first.chunk["type"] == "snapshot"
                await queue_event(manager)
                update = await asyncio.wait_for(queue.get(), 2)
                assert update.chunk_type == "data" and update.chunk["type"] == "update"
                manager.overview_events.close()
                assert (await asyncio.wait_for(queue.get(), 2)).chunk_type == "done"
                await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            task = asyncio.create_task(job(stream_queue=queue, _allowed_paths=["resource"]))
            assert (await asyncio.wait_for(queue.get(), 2)).chunk_type == "error"
            assert (await asyncio.wait_for(queue.get(), 2)).chunk_type == "done"
            await task
        assert not wrapper.created and not wrapper.turns
        assert not manager.overview_events.listeners


@pytest.mark.asyncio
async def test_heartbeat_shutdown_and_stream_job_terminal_contract(tmp_path):
    async with running_system(tmp_path) as (manager, context, _):
        events = overview_views(manager, context).events(interval=0.01, heartbeat=0.01)
        await anext(events)
        assert (await anext(events)) == {"type": "heartbeat"}
        manager.overview_events.close()
        with pytest.raises(StopAsyncIteration):
            await anext(events)
        assert not manager.overview_events.listeners
        context.components.setdefault(
            "connector_manager",
            {"default": ConnectorManager(app_context=context, agent_manager=manager)},
        )
        config = context.app_config.jobs["watch_proactive_overview"].model_dump(exclude={"backend"})
        job = StreamJob(name="watch_proactive_overview", app_context=context, **config)
        queue = asyncio.Queue()
        async with job:
            await job(stream_queue=queue, agent_names=[])
        chunks = [queue.get_nowait() for _ in range(queue.qsize())]
        assert [chunk.chunk_type for chunk in chunks] == ["error", "done"]
        assert job.stream_queue_size == 1


@pytest.mark.asyncio
async def test_source_collection_and_admission_are_independent_durable_facts(tmp_path, monkeypatch):
    class Collector(BaseConnectorStep):
        """Write local fixture data instead of contacting a provider."""

        source = "overview_fixture"

        def collect(self, directory):
            (directory / "data.txt").write_text("observed")
            return [{"path": "data.txt"}]

    async with running_system(tmp_path, runtimes=[{"agent_name": "monitor", "agent_type": "event"}]) as (
        manager,
        context,
        _,
    ):
        context.registry.add("overview_fixture_step", Collector, owner=__name__)
        config = JobConfig(
            backend="base",
            steps=[
                {
                    "backend": "overview_fixture_step",
                    "connector_job": "overview_collect",
                    "runtime_names": ["monitor"],
                },
            ],
        )
        context.app_config.jobs["overview_collect"] = config
        job = BaseJob(name="overview_collect", app_context=context, **config.model_dump(exclude={"backend"}))
        context.jobs[job.name] = job
        async with job:
            assert (await job()).success
            views = overview_views(manager, context)
            value = await views.read()
            source = next(row for row in value["sources"] if row["source_id"] == "preset:overview_collect")
            assert source["last_collected_at"] and source["last_admitted_at"]
            assert source["pending_delivery_count"] == 0
            original = manager.deliver_events

            async def blocked(*_args, **_kwargs):
                return {"deliveries": {}, "rejected": {}, "errors": {"monitor": "offline"}}

            monkeypatch.setattr(manager, "deliver_events", blocked)
            assert not (await job()).success
            value = await views.read()
            changed = next(row for row in value["sources"] if row["source_id"] == "preset:overview_collect")
            assert changed["last_admitted_at"] == source["last_admitted_at"]
            assert changed["pending_delivery_count"] == 1 and "offline" in changed["last_error"]
            monkeypatch.setattr(manager, "deliver_events", original)
            assert (await job()).success
            value = await views.read()
            changed = next(row for row in value["sources"] if row["source_id"] == "preset:overview_collect")
            assert changed["pending_delivery_count"] == 0 and changed["last_error"] is None
            marker = next((tmp_path / "resource" / Collector.source).glob("*/*/.collection.json"))
            marker.write_text("invalid JSON")
            value = await views.read()
            assert "source:preset:overview_collect" in value["errors"]
            assert any(row["source_id"] == "preset:overview_collect" and row["last_error"] for row in value["sources"])


def test_request_validation_and_bounded_journal():
    for value in (
        {"agent_names": []},
        {"agent_names": ["one", "ONE"]},
        {"agent_names": ["Remy"]},
        {"path": "../escape"},
    ):
        with pytest.raises((ValidationError, ValueError)):
            WatchProactiveOverviewRequest.model_validate(value)
    journal = OverviewEvents(capacity=2)
    old_epoch = journal.epoch
    for index in range(10):
        journal.publish("event_admitted", identity=str(index), agent_name="one", event_ids=[str(index)])
    assert len(journal.buffer) == 2 and len(journal.identities) <= 8
    assert journal.since(0) is None
    journal.reset()
    assert journal.epoch != old_epoch and journal.revision == 0
