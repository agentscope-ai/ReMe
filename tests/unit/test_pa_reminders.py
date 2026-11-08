"""Round binding, last-decision state and cancellation-safe deferred delivery."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.steps.proactive.tools import GetChatMessagesStep, SendChatMessageStep


async def submit(env, content):
    async with env.runtime.proactive_round() as current:
        current.phase = "query_message"
        return await env.runtime.send_chat_message(
            content,
            kind="normal",
            round_id=current.id,
            session_id=current.session_id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase",
    [
        None,
        "query_memory",
        "query_history",
        "query_tasks",
        "query_research",
        "query_config",
        "query_todo",
        "query_reminder",
        "unknown_phase",
    ],
)
@pytest.mark.parametrize("content", ["", "不应提前发送。"])
async def test_other_phases_cannot_submit_or_consume_the_reminder_decision(pa_workspace, phase, content):
    env = pa_workspace
    async with env.runtime.proactive_round() as current:
        current.phase = phase
        with pytest.raises(ValueError, match="only allowed during query_message"):
            await SendChatMessageStep(app_context=env.context)(
                content=content,
                pa_caller="proactive",
                kind="normal",
                pa_round_id=current.id,
                session_id=current.session_id,
            )
        assert current.request is None and await env.runtime.get_chat_messages() == []
        env.agent.append_session.assert_not_awaited()
        current.phase = "query_message"
        result = await env.runtime.send_chat_message(
            "有用的新结果。",
            kind="normal",
            round_id=current.id,
            session_id=current.session_id,
        )
        assert result["status"] == "pending" and len(await env.runtime.get_chat_messages()) == 1
        env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_requires_current_bound_round_and_rejects_stale_tokens(pa_workspace):
    env = pa_workspace
    with pytest.raises(ValueError, match="live, server-bound"):
        await SendChatMessageStep(app_context=env.context)(content="hello", pa_caller="proactive")
    async with env.runtime.proactive_round() as current:
        old = current.id
        with pytest.raises(ValueError, match="live, server-bound"):
            await env.runtime.send_chat_message("hello", kind="normal", round_id=current.id, session_id="another")
    async with env.runtime.proactive_round():
        with pytest.raises(ValueError, match="live, server-bound"):
            await env.runtime.send_chat_message("hello", kind="normal", round_id=old, session_id="session-1")
    assert await env.runtime.get_chat_messages() == []


@pytest.mark.asyncio
async def test_concurrent_retries_in_one_round_inject_only_once(pa_workspace):
    env = pa_workspace
    async with env.runtime.proactive_round() as current:
        current.phase = "query_message"

        async def send():
            return await env.runtime.send_chat_message(
                "结果有价值。",
                kind="normal",
                round_id=current.id,
                session_id=current.session_id,
            )

        first, second = await asyncio.gather(send(), send())
        assert first["id"] == second["id"] and first["status"] == second["status"] == "pending"
        env.agent.append_session.assert_not_awaited()
        with pytest.raises(ValueError, match="Only one"):
            await env.runtime.send_chat_message(
                "different",
                kind="normal",
                round_id=current.id,
                session_id=current.session_id,
            )
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_awaited_once()
    assert len(await env.runtime.get_chat_messages()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["idle", "not_loaded"])
async def test_already_idle_chat_still_requires_ten_seconds(pa_workspace, status):
    env = pa_workspace
    env.snapshot.state.status = status
    result = await submit(env, "已经空闲也要等待。")
    assert result["status"] == "pending" and env.runtime.reminders.idle is not None
    env.agent.append_session.assert_not_awaited()
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 9)
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_not_awaited()
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_awaited_once()
    assert (await env.runtime.get_chat_messages())[0]["status"] == "sent"


@pytest.mark.asyncio
async def test_changed_turn_resets_idle_delay_even_if_chat_is_idle(pa_workspace):
    env = pa_workspace
    await submit(env, "新轮次要重新计时。")
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    env.snapshot.last_turn_id = "turn-2"
    await env.runtime.deliver_pending_reminder()
    assert env.runtime.reminders.idle[0] == "turn-2"
    env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_delivery_outlives_round_and_waits_ten_idle_seconds(pa_workspace):
    env = pa_workspace
    env.snapshot.state.status = "active"
    result = await submit(env, "等空闲再说。")
    assert result["status"] == "pending" and env.runtime.current_round is None
    env.snapshot.state.status = "idle"
    env.snapshot.last_turn_id = "turn-2"
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_not_awaited()
    env.runtime.reminders.idle = ("turn-2", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_awaited_once()
    assert (await env.runtime.get_chat_messages())[0]["status"] == "sent"


@pytest.mark.asyncio
async def test_new_activity_resets_idle_delay(pa_workspace):
    env = pa_workspace
    env.snapshot.state.status = "active"
    await submit(env, "稍后提醒。")
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    assert env.runtime.reminders.idle is None
    env.snapshot.state.status = "idle"
    env.snapshot.last_turn_id = "turn-2"
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_decision_replaces_pending_and_empty_decision_withdraws_it(pa_workspace):
    env = pa_workspace
    env.snapshot.state.status = "active"
    first = await submit(env, "旧提醒。")
    second = await submit(env, "新提醒。")
    records = {row["id"]: row for row in await env.runtime.get_chat_messages()}
    assert records[first["id"]]["status"] == "discarded" and records[second["id"]]["status"] == "pending"
    assert records[first["id"]]["reason"] == "superseded"
    assert (await submit(env, "  "))["status"] == "skipped"
    assert all(row["status"] == "discarded" for row in await env.runtime.get_chat_messages())
    assert (await env.runtime.get_chat_messages())[0]["reason"] == "withdrawn"
    env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_decision_after_restart_withdraws_pending_but_preserves_sent(pa_workspace):
    env = pa_workspace
    first = await submit(env, "已经发出的消息。")
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    pending = await submit(env, "尚未发出的消息。")
    env.runtime.reminders.restore()
    await submit(env, "")
    env.runtime.reminders.restore()
    records = {row["id"]: row for row in await env.runtime.get_chat_messages()}
    assert records[first["id"]]["status"] == "sent"
    assert records[pending["id"]]["status"] == "discarded"
    assert records[pending["id"]]["reason"] == "withdrawn"
    assert env.runtime.reminders.pending is None
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "condition, reason",
    [("session", "session_changed"), ("expired", "expired"), ("hours", "outside_active_hours")],
)
async def test_stale_pending_content_is_discarded(pa_workspace, condition, reason):
    env = pa_workspace
    env.snapshot.state.status = "active"
    await submit(env, "待评估内容。")
    if condition == "session":
        env.runtime.session_id = "new-session"
    elif condition == "expired":
        env.now[0] += timedelta(minutes=6)
    else:
        env.now[0] = env.now[0].replace(hour=23)
    await env.runtime.deliver_pending_reminder()
    record = (await env.runtime.get_chat_messages())[0]
    assert record["status"] == "discarded" and record["reason"] == reason
    env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_submission_finishes_idle_check_and_status_publication(pa_workspace):
    env = pa_workspace
    entered, release = asyncio.Event(), asyncio.Event()

    async def read(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return env.snapshot

    env.agent.read_session = AsyncMock(side_effect=read)
    async with env.runtime.proactive_round() as current:
        current.phase = "query_message"
        task = asyncio.create_task(
            env.runtime.send_chat_message(
                "确定结果。",
                kind="normal",
                round_id=current.id,
                session_id=current.session_id,
            ),
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert (await env.runtime.get_chat_messages())[0]["status"] == "pending"
    env.agent.append_session.assert_not_awaited()
    assert not env.runtime.reminders.lock.locked() and not env.runtime.session_lock.locked()
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    assert (await env.runtime.get_chat_messages())[0]["status"] == "sent"


@pytest.mark.asyncio
async def test_restart_rebuilds_pending_and_history_is_read_only(pa_workspace):
    env = pa_workspace
    env.snapshot.state.status = "active"
    await submit(env, "可恢复提醒。")
    original = env.runtime.chat_inbox_path.read_bytes()
    env.runtime.reminders.pending = None
    env.runtime.reminders.restore()
    assert env.runtime.reminders.pending is not None
    result = await GetChatMessagesStep(app_context=env.context)()
    assert result.answer[0]["status"] == "pending" and env.runtime.chat_inbox_path.read_bytes() == original
    assert result.answer[0]["content"] == "可恢复提醒。"
    assert "history_path" not in result.answer[0]
