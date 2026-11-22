"""Regression cases reported while reviewing reminder scheduling and file ownership."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio
import gc
import weakref
from datetime import datetime
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.schema import Event, Response, SessionMessage
from reme.steps.file_io import SaveStep, get_path_lock
from .test_proactive_runtime import completed_result, running_runtime
from .test_inbox_fork import prepare


@pytest.mark.asyncio
async def test_path_lock_survives_handoff_and_reclaims_unused_entries(tmp_path):
    target = tmp_path / "target.md"
    lock = await get_path_lock(target)
    await lock.acquire()
    entered, release = asyncio.Event(), asyncio.Event()

    async def waiting_writer():
        async with await get_path_lock(target):
            entered.set()
            await release.wait()

    waiter = asyncio.create_task(waiting_writer())
    try:
        await asyncio.sleep(0)
        others = [await get_path_lock(tmp_path / str(index)) for index in range(1024)]
        lock.release()
        # The waiter has been woken, but has not resumed to set locked() yet.
        others.append(await get_path_lock(tmp_path / "overflow"))
        current = await get_path_lock(target)
        assert current is lock
        await asyncio.wait_for(entered.wait(), 2)
        assert current.locked()
    finally:
        release.set()
        await asyncio.wait_for(waiter, 2)
    reference = weakref.ref(current)
    del current, lock
    gc.collect()
    assert reference() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["disable", "delete", "window"])
async def test_ineligible_reminders_release_capacity_and_resume_without_losing_unread_state(tmp_path, action):
    async with running_runtime(tmp_path) as (runtime, _, _):
        for name, priority in (("first", "P2"), ("second", "P0")):
            await runtime.create(
                {
                    "agent_name": name,
                    "task_description": "Report",
                },
            )
            await runtime.append_events(
                name,
                [Event(event_ts=runtime.now(), source="test", description="Finished")],
            )
            result, session = completed_result(runtime, name, priority=priority)
            await runtime.run(name, AsyncMock(return_value=(result, session)))
            if name == "first":
                await runtime.outbox.sync()
                if action == "delete":
                    await runtime.delete(name)
                elif action == "disable":
                    await runtime.update(name, {"enabled": False})
                else:
                    hour = (runtime.now().hour + 1) % 24
                    await runtime.update(
                        name,
                        {
                            "notify_window": {
                                "start": f"{hour:02}:00",
                                "end": f"{hour:02}:01",
                            },
                        },
                    )
        runtime.inbox.max_pending_items = 1
        runtime.notifications.config.p0_delay_seconds = 0
        handled = []

        async def decide(_name, *, priority, msg_ids):
            assert priority == "P0"
            handled.extend(msg_ids)
            await runtime.notifications.consume(msg_ids)
            return Response(answer={"sent": False})

        async with asyncio.timeout(3):
            for _ in range(5):
                await runtime.notifications.tick(decide)
                if worker := runtime.notifications.worker:
                    await worker
        assert len(handled) == 1
        first = next(ref for ref in runtime.outbox.records if runtime.outbox.owners[ref] == "first")
        assert runtime.outbox.records[first].read_at is None
        assert "read_at" not in frontmatter.loads(runtime.files.path(first).read_text())
        assert not runtime.notifications.active_turn
        if action == "delete":
            await runtime.create(
                {
                    "agent_name": "first",
                    "task_description": "Restore",
                },
            )
        else:
            await runtime.update("first", {"enabled": True, "notify_window": None})
        await runtime.outbox.sync()
        assert first in runtime.notifications.pending
        assert runtime.outbox.records[first].read_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["The experiment finished.", ""])
@pytest.mark.parametrize("changed_field", ["reminder_message", "reminder_priority"])
async def test_decision_revalidates_after_a_queued_editor_save(tmp_path, monkeypatch, content, changed_field):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        args = await prepare(runtime, context)
        source = (await runtime.sessions.current("Remy")).session_id
        ref = args["msg_ids"][0]
        path = runtime.files.path(ref)
        document = frontmatter.loads(path.read_text())
        document[changed_field] = "P2" if changed_field == "reminder_priority" else "The experiment was cancelled."
        wrapper.append_session = AsyncMock()
        refreshed = asyncio.Event()
        refresh = runtime.outbox.refresh

        async def observe_refresh():
            result = await refresh()
            refreshed.set()
            return result

        monkeypatch.setattr(runtime.outbox, "refresh", observe_refresh)
        async with runtime.notifications.turn(source, "P1", args["msg_ids"]) as turn:
            await runtime.notifications.read_turn(turn)
            async with await get_path_lock(path):
                edit = asyncio.create_task(
                    SaveStep(app_context=context)(
                        path=ref,
                        content=frontmatter.dumps(document),
                        expected_mtime=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    ),
                )
                await asyncio.sleep(0)
                decision = asyncio.create_task(runtime.notifications.send(turn, content))
                await asyncio.wait_for(refreshed.wait(), 2)
                await asyncio.sleep(0)
            assert (await asyncio.wait_for(edit, 2)).success
            with pytest.raises(RuntimeError, match="reminders changed"):
                await asyncio.wait_for(decision, 2)
        wrapper.append_session.assert_not_awaited()
        latest = frontmatter.loads(path.read_text())
        assert latest[changed_field] == document[changed_field]
        assert "read_at" not in latest


@pytest.mark.asyncio
async def test_editor_cannot_save_between_native_delivery_and_acknowledgement(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, wrapper):
        args = await prepare(runtime, context)
        source = (await runtime.sessions.current("Remy")).session_id
        ref = args["msg_ids"][0]
        path = runtime.files.path(ref)
        entered, release = asyncio.Event(), asyncio.Event()

        async def append(_session, content, *, message_id, **_kwargs):
            entered.set()
            await release.wait()
            return SessionMessage(id=message_id, role="assistant", content=content)

        wrapper.append_session = AsyncMock(side_effect=append)
        async with runtime.notifications.turn(source, "P1", args["msg_ids"]) as turn:
            await runtime.notifications.read_turn(turn)
            decision = asyncio.create_task(runtime.notifications.send(turn, "The experiment finished."))
            edit = None
            try:
                await asyncio.wait_for(entered.wait(), 2)
                document = frontmatter.loads(path.read_text())
                document.content = "New user content"
                edit = asyncio.create_task(
                    SaveStep(app_context=context)(
                        path=ref,
                        content=frontmatter.dumps(document),
                        expected_mtime=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    ),
                )
                await asyncio.sleep(0.02)
                assert not edit.done()
            finally:
                release.set()
                await asyncio.wait_for(decision, 2)
            response = await asyncio.wait_for(edit, 2)
            assert not response.success and response.metadata["code"] == "file_conflict"
        assert not runtime.notifications.pending
        assert frontmatter.loads(path.read_text())["read_at"] is not None


@pytest.mark.asyncio
async def test_reenabled_task_waits_for_capacity_without_blocking_an_urgent_reminder(tmp_path):
    async with running_runtime(tmp_path) as (runtime, _, _):
        for name, priority in (("first", "P2"), ("second", "P0")):
            await runtime.create({"agent_name": name, "task_description": "Report"})
            await runtime.append_events(name, [Event(event_ts=runtime.now(), source="test", description="Finished")])
            result, session = completed_result(runtime, name, priority=priority)
            await runtime.run(name, AsyncMock(return_value=(result, session)))
            if name == "first":
                await runtime.outbox.sync()
                await runtime.update(name, {"enabled": False})
        runtime.inbox.max_pending_items = 1
        await runtime.outbox.sync()
        urgent = next(ref for ref in runtime.outbox.records if runtime.outbox.owners[ref] == "second")
        await runtime.update("first", {"enabled": True})
        await runtime.outbox.sync()
        assert urgent in runtime.notifications.pending
        await runtime.notifications.consume([urgent])
        first = next(ref for ref in runtime.outbox.records if runtime.outbox.owners[ref] == "first")
        assert first in runtime.notifications.pending
        assert runtime.outbox.records[first].read_at is None


@pytest.mark.asyncio
async def test_native_decision_holds_file_lock_against_other_writers(tmp_path):
    async with running_runtime(tmp_path) as (runtime, context, _):
        args = await prepare(runtime, context)
        ref = args["msg_ids"][0]
        items = (await runtime.notifications.read([ref]))["items"]

        async with runtime.outbox.decision(items) as update:
            other = asyncio.create_task(runtime.outbox.update(ref, read_at=runtime.now()))
            await asyncio.sleep(0)
            assert not other.done()
            await update(ref, chat_message_ids=["reme-proactive-example"])

        await asyncio.wait_for(other, 2)
        item = runtime.outbox.records[ref]
        assert item.read_at and item.chat_message_ids == ["reme-proactive-example"]
