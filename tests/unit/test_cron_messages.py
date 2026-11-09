"""Single-query cron submissions retain fresh-history and retry-safe delivery guards."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.steps.proactive import CronAgentStep
from reme.steps.proactive.tools import GetChatMessagesStep, SendChatMessageImmediateStep
from .pa_helpers import configure, stub_model, task


async def history(env, binding, **kwargs):
    return await GetChatMessagesStep(app_context=env.context)(**binding, **kwargs)


async def submit(env, binding, reminder="主人，邮件中的确认期限马上到了。"):
    return await SendChatMessageImmediateStep(app_context=env.context)(**binding, content=reminder)


async def other_message(env):
    temporary = await env.runtime.temp_tasks.create("明确约定的提醒")
    await env.runtime.temp_tasks.patch(temporary.id, temporary.revision, status="running")
    await env.runtime.temp_tasks.send(temporary.id, temporary.revision, "另一个 Agent 的新提醒")


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["主人，确认期限马上到了。", "完整发现和依据。" * 2000], ids=["short", "long"])
async def test_submission_requires_history_publishes_before_send_and_retries_once(pa_workspace, monkeypatch, content):
    env = pa_workspace
    bindings = []

    async def message(binding):
        bindings.append(binding)
        with pytest.raises(ValueError, match="get_chat_messages"):
            await submit(env, binding, reminder=content)
        assert env.runtime.tasks.paths("one") == [] and env.runtime.reminders.read() == []
        await history(env, binding)
        first, second = await submit(env, binding, reminder=content), await submit(env, binding, reminder=content)
        assert first.answer == second.answer

    stub_model(env, monkeypatch, message=message)
    result = await CronAgentStep(app_context=env.context)(task=task())
    record = env.runtime.reminders.read()[0]
    assert record.content == content
    assert record.result_path == result.answer["path"] and record.session_id == "session-1"
    assert record.expires_at == env.now[0] + timedelta(minutes=15)
    assert len(env.runtime.reminders.read()) == 1 and env.runtime.tasks.executions == {}
    assert (
        frontmatter.load(env.runtime.workspace_path / record.result_path)["started_at"] == result.answer["started_at"]
    )
    with pytest.raises(ValueError, match="current execution"):
        await submit(env, bindings[0])


@pytest.mark.asyncio
async def test_history_race_requires_reconsideration(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        await history(env, binding)
        await other_message(env)
        with pytest.raises(ValueError, match="history changed"):
            await submit(env, binding)
        assert len(env.runtime.reminders.read()) == 1
        assert (await history(env, binding)).answer[0]["content"] == "另一个 Agent 的新提醒"
        await submit(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    assert len(env.runtime.reminders.read()) == 2


@pytest.mark.asyncio
async def test_cancelled_definition_cannot_publish_or_send(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        await history(env, binding)
        configure(env, [])
        await submit(env, binding)

    stub_model(env, monkeypatch, message=message)
    with pytest.raises(ValueError, match="changed or was cancelled"):
        await CronAgentStep(app_context=env.context)(task=task())
    assert env.runtime.reminders.read() == [] and env.runtime.tasks.paths("one") == []
    assert env.runtime.tasks.executions == {}


@pytest.mark.asyncio
async def test_cold_recovery_after_receipt_does_not_repeat_model_or_message(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        await history(env, binding)
        await submit(env, binding)

    stub_model(env, monkeypatch, message=message)
    original = env.runtime.reminders.submit_immediate

    async def crash(*args, **kwargs):
        await original(*args, **kwargs)
        raise asyncio.CancelledError()

    monkeypatch.setattr(env.runtime.reminders, "submit_immediate", crash)
    with pytest.raises(asyncio.CancelledError):
        await CronAgentStep(app_context=env.context)(task=task())
    record = env.runtime.reminders.read()[0]
    path = env.runtime.tasks.paths("one")[0]
    monkeypatch.setattr(env.runtime.reminders, "submit_immediate", original)
    env.agent.reply = AsyncMock(side_effect=AssertionError("Already submitted"))
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert result.answer["status"] == "succeeded" and result.answer["message_id"] == record.id
    assert env.runtime.tasks.paths("one") == [path]
    assert [r.id for r in env.runtime.reminders.read()] == [record.id]
    env.agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_recovered_candidate_is_dropped_without_another_query(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        await history(env, binding)
        await submit(env, binding)

    stub_model(env, monkeypatch, message=message)
    publish = env.runtime.tasks._publish
    monkeypatch.setattr(env.runtime.tasks, "_publish", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        await CronAgentStep(app_context=env.context)(task=task())
    await other_message(env)
    monkeypatch.setattr(env.runtime.tasks, "_publish", publish)
    env.agent.reply = AsyncMock(side_effect=AssertionError("At most one query per run"))
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert result.answer["status"] == "succeeded" and result.answer["reason"] == "stale_reminder"
    assert len(env.runtime.tasks.paths("one")) == 1 and len(env.runtime.reminders.read()) == 1
    env.agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_urgent_submission_bypasses_pause_hours_and_keeps_original_main(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        env.now[0] = env.now[0].replace(hour=23, minute=30)
        config = await env.runtime.get_proactive_config()
        env.runtime.files.write_json(
            env.runtime.proactive_config_path,
            {**config.model_dump(mode="json"), "enabled": False},
        )
        env.runtime.session_id = "session-2"
        await history(env, binding)
        await submit(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    record = env.runtime.reminders.read()[0]
    assert record.delivery == "immediate" and record.announced_at is not None and record.session_id == "session-1"


@pytest.mark.asyncio
async def test_expired_urgency_is_not_delivered_later(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        await history(env, binding)
        await submit(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    env.now[0] += timedelta(minutes=16)
    await env.runtime.deliver_pending_reminder()
    record = env.runtime.reminders.read()[0]
    assert (record.status, record.reason) == ("discarded", "expired")
    env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_main_still_publishes_without_reminder(pa_workspace, monkeypatch):
    env = pa_workspace
    env.runtime.session_id = None
    stub_model(env, monkeypatch)
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert result.answer["status"] == "succeeded" and env.runtime.tasks.paths("one")
    env.agent.reply.assert_awaited_once()
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_history_pagination_does_not_mix_snapshots(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        await history(env, binding)
        await other_message(env)
        with pytest.raises(ValueError, match="offset=0"):
            await history(env, binding, offset=1)
        await history(env, binding)
        await submit(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())


@pytest.mark.asyncio
async def test_submission_and_history_preserve_permission_scope(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        with pytest.raises(PermissionError):
            await submit(env, {**binding, "_allowed_paths": []})
        with pytest.raises(PermissionError):
            await history(env, {**binding, "_allowed_paths": []})
        await history(env, binding)
        await submit(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())


@pytest.mark.asyncio
async def test_wrong_task_cannot_submit_in_reused_session(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        with pytest.raises(ValueError, match="current execution"):
            await submit(env, {**binding, "cron_task_id": "different"})
        await history(env, binding)
        await submit(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    await CronAgentStep(app_context=env.context)(task=task())
    env.agent.create_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_revoked_run_cannot_commit_queued_reminder(pa_workspace, monkeypatch):
    env = pa_workspace

    async def message(binding):
        await history(env, binding)
        async with env.runtime.reminders.lock:
            queued = asyncio.create_task(submit(env, binding))
            async with asyncio.timeout(3):
                while not env.runtime.tasks.paths("one"):
                    await asyncio.sleep(0.005)
            env.runtime.tasks.executions.pop(binding["cron_task_id"])
        with pytest.raises(ValueError, match="current execution"):
            await queued
        raise RuntimeError("revoked")

    stub_model(env, monkeypatch, message=message)
    with pytest.raises(RuntimeError, match="revoked"):
        await CronAgentStep(app_context=env.context)(task=task())
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", [None, "chat", "proactive"])
async def test_immediate_tool_rejects_unbound_callers(pa_workspace, caller):
    with pytest.raises(ValueError, match="server-bound"):
        await SendChatMessageImmediateStep(app_context=pa_workspace.context)(pa_caller=caller, content="x")
