"""Round binding, last-decision state and cancellation-safe deferred delivery."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from reme.steps.proactive.tools import GetReminderStep, SendReminderStep


async def submit(env, content):
    async with env.runtime.proactive_round() as current:
        return await env.runtime.send_reminder(
            content,
            kind="normal",
            round_id=current.id,
            session_id=current.session_id,
        )


@pytest.mark.asyncio
async def test_requires_current_bound_round_and_rejects_stale_tokens(pa_workspace):
    env = pa_workspace
    with pytest.raises(ValueError, match="live, server-bound"):
        await SendReminderStep(app_context=env.context)(content="hello")
    async with env.runtime.proactive_round() as current:
        old = current.id
        with pytest.raises(ValueError, match="live, server-bound"):
            await env.runtime.send_reminder("hello", kind="normal", round_id=current.id, session_id="another")
    async with env.runtime.proactive_round():
        with pytest.raises(ValueError, match="live, server-bound"):
            await env.runtime.send_reminder("hello", kind="normal", round_id=old, session_id="session-1")
    assert await env.runtime.get_reminder() == []


@pytest.mark.asyncio
async def test_concurrent_retries_in_one_round_inject_only_once(pa_workspace):
    env = pa_workspace
    async with env.runtime.proactive_round() as current:

        async def send():
            return await env.runtime.send_reminder(
                "结果有价值。",
                kind="normal",
                round_id=current.id,
                session_id=current.session_id,
            )

        first, second = await asyncio.gather(send(), send())
        assert first["id"] == second["id"] and first["status"] == second["status"] == "sent"
        with pytest.raises(ValueError, match="Only one"):
            await env.runtime.send_reminder(
                "different",
                kind="normal",
                round_id=current.id,
                session_id=current.session_id,
            )
    env.agent.append_session.assert_awaited_once()
    assert len(await env.runtime.get_reminder()) == 1


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
    assert (await env.runtime.get_reminder())[0]["status"] == "sent"


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
    records = {row["id"]: row for row in await env.runtime.get_reminder()}
    assert records[first["id"]]["status"] == "discarded" and records[second["id"]]["status"] == "pending"
    assert (await submit(env, "  "))["status"] == "skipped"
    assert all(row["status"] == "discarded" for row in await env.runtime.get_reminder())
    env.agent.append_session.assert_not_awaited()


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
    record = (await env.runtime.get_reminder())[0]
    assert record["status"] == "discarded" and record["reason"] == reason
    env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_submission_finishes_injection_and_status_publication(pa_workspace):
    env = pa_workspace
    entered, release = asyncio.Event(), asyncio.Event()

    async def append(*_args, **_kwargs):
        entered.set()
        await release.wait()

    env.agent.append_session = AsyncMock(side_effect=append)
    async with env.runtime.proactive_round() as current:
        task = asyncio.create_task(
            env.runtime.send_reminder("确定结果。", kind="normal", round_id=current.id, session_id=current.session_id),
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert (await env.runtime.get_reminder())[0]["status"] == "sent"
    assert not env.runtime.reminders.lock.locked() and not env.runtime.session_lock.locked()


@pytest.mark.asyncio
async def test_restart_rebuilds_pending_and_history_is_read_only(pa_workspace):
    env = pa_workspace
    env.snapshot.state.status = "active"
    await submit(env, "可恢复提醒。")
    original = env.runtime.reminders_path.read_bytes()
    env.runtime.reminders.pending = None
    env.runtime.reminders.restore()
    assert env.runtime.reminders.pending is not None
    result = await GetReminderStep(app_context=env.context)()
    assert result.answer[0]["status"] == "pending" and env.runtime.reminders_path.read_bytes() == original
