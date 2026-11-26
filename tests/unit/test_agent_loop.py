"""Notification scheduling is owned by a Background Job."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.core.context import RuntimeContext
from reme.schema import Response
from reme.steps.proactive.chat import ChatNotificationsLoopStep
from .proactive_helpers import running_system, emit_notification


@pytest.mark.parametrize("priority,delay", [("P0", 0), ("P1", 10), ("P2", 1800)])
@pytest.mark.asyncio
async def test_priority_idle_delays_and_busy_foreground(tmp_path, monkeypatch, priority, delay):
    async with running_system(tmp_path) as (manager, context, _):
        ref = await emit_notification(manager, context, priority=priority)
        chat = context.components["agent_runtime"]["Remy"]
        now = chat.now()
        monkeypatch.setattr(chat, "now", lambda: now)
        loop = ChatNotificationsLoopStep(app_context=context, agent_runtime="Remy")
        loop.context = RuntimeContext()
        called = AsyncMock(return_value=Response(answer={"sent": False}))
        monkeypatch.setattr(loop, "run_job", called)
        chat.last_activity_at = now - timedelta(seconds=max(0, delay - 0.1))
        if delay:
            await loop.tick()
            assert chat.notification_worker is None
        chat.last_activity_at = now - timedelta(seconds=delay)
        await loop.tick()
        worker = chat.notification_worker
        assert worker is not None
        await worker
        called.assert_awaited_once_with("process_chat_notifications", priority=priority, msg_ids=[ref.id])


@pytest.mark.asyncio
async def test_priority_batching_skips_oversized_and_missing_records(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, context, _):
        await emit_notification(manager, context, suffix=1, priority="P0", message="x" * 2000)
        missing = await emit_notification(manager, context, suffix=2, priority="P0")
        (tmp_path / missing.record_path).unlink()
        selected = await emit_notification(manager, context, suffix=3, priority="P0")
        chat = context.components["agent_runtime"]["Remy"]
        chat.inbox.claim_max_bytes = 1000
        loop = ChatNotificationsLoopStep(app_context=context, agent_runtime="Remy")
        loop.context = RuntimeContext()
        call = AsyncMock(return_value=Response())
        monkeypatch.setattr(loop, "run_job", call)
        await loop.tick()
        await chat.notification_worker
        assert call.await_args.kwargs["msg_ids"] == [selected.id]


@pytest.mark.asyncio
async def test_background_stop_cancels_and_joins_notification_worker(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, context, _):
        await emit_notification(manager, context, priority="P0")
        entered = asyncio.Event()

        async def slow(*_, **__):
            entered.set()
            await asyncio.Event().wait()

        loop = ChatNotificationsLoopStep(app_context=context, agent_runtime="Remy")
        monkeypatch.setattr(loop, "run_job", slow)
        stop = asyncio.Event()
        task = asyncio.create_task(loop(RuntimeContext(stop_event=stop)))
        await asyncio.wait_for(entered.wait(), 1)
        worker = context.components["agent_runtime"]["Remy"].notification_worker
        stop.set()
        await asyncio.wait_for(task, 2)
        assert worker.cancelled()
