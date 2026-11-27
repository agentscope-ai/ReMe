"""Scheduler health, isolated missing records and bounded notification retries."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.core.context import RuntimeContext
from reme.schema import Response
from reme.steps.proactive.chat import ChatNotificationsLoopStep
from reme.steps.proactive.agent_api import AgentSessionStep, AgentViews
from .proactive_helpers import running_system, emit_notification


@pytest.mark.parametrize("permanent", [False, True])
@pytest.mark.asyncio
async def test_notification_failure_backs_off_or_blocks_incompatible_sdk(tmp_path, monkeypatch, permanent):
    async with running_system(tmp_path) as (manager, context, _):
        await emit_notification(manager, context, priority="P0")
        chat = context.components["agent_runtime"]["Remy"]
        now = chat.now()
        monkeypatch.setattr(chat, "now", lambda: now)
        response = Response(
            success=False,
            answer="SDK mismatch" if permanent else "Temporary network error",
            metadata={"reme.job_error": {"type": "TypeError" if permanent else "ConnectionError"}},
        )
        call = AsyncMock(return_value=response)
        loop = ChatNotificationsLoopStep(app_context=context, agent_runtime="Remy")
        loop.context = RuntimeContext()
        monkeypatch.setattr(loop, "run_job", call)
        await loop.tick()
        await chat.notification_worker
        assert chat.blocked is permanent and chat.retry_at == now + timedelta(seconds=5)
        views = AgentViews(AgentSessionStep(app_context=context, agent_runtime="Remy"))
        assert (await views.view("Remy")).status == "error"
        await loop.tick()
        assert call.await_count == 1
        now += timedelta(seconds=5)
        call.return_value = Response()
        await loop.tick()
        if permanent:
            assert call.await_count == 1
        else:
            await chat.notification_worker
            assert call.await_count == 2 and chat.retry_at is None and chat.error is None


@pytest.mark.asyncio
async def test_missing_record_is_reported_without_erasing_refs_and_restores_after_restart(tmp_path, monkeypatch):
    async with running_system(tmp_path) as (manager, context, _):
        missing = await emit_notification(manager, context, priority="P0")
        path = tmp_path / missing.record_path
        saved = path.read_text()
        path.unlink()
        okay = await emit_notification(manager, context, suffix=1, priority="P0")
        loop = ChatNotificationsLoopStep(app_context=context, agent_runtime="Remy")
        loop.context = RuntimeContext()
        call = AsyncMock(return_value=Response())
        monkeypatch.setattr(loop, "run_job", call)
        await loop.tick()
        await context.components["agent_runtime"]["Remy"].notification_worker
        assert call.await_args.kwargs["msg_ids"] == [okay.id]
    path.write_text(saved)
    async with running_system(tmp_path) as (_, context, _):
        chat = context.components["agent_runtime"]["Remy"]
        assert await chat.inbox.pending_ids() == {missing.id, okay.id}
        assert (await chat.reminder_item(await chat.inbox.reference(missing.id)))["message"] == "Verified result"
