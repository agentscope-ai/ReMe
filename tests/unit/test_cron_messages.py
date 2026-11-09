"""Published cron decisions, history races and durable immediate delivery."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.steps.proactive import CronAgentStep
from reme.steps.proactive.tools import GetChatMessagesStep, SendChatMessageImmediateStep

from .test_pa_cron import configure, stub_model, task


async def history(env, binding):
    return await GetChatMessagesStep(app_context=env.context)(**binding)


async def send(env, binding, content="主人，邮件中的确认期限马上到了。"):
    return await SendChatMessageImmediateStep(app_context=env.context)(**binding, content=content)


@pytest.mark.asyncio
async def test_cron_sends_only_after_publication_and_current_history(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    bindings, receipts = [], []

    async def message(binding):
        bindings.append(binding)
        published = env.runtime.tasks.paths("one")
        assert len(published) == 1 and "本次发现" in frontmatter.load(published[0]).content
        with pytest.raises(ValueError, match="get_chat_messages"):
            await send(env, binding)
        assert env.runtime.reminders.read() == []
        await history(env, binding)
        receipts.append((await send(env, binding)).answer)
        receipts.append((await send(env, binding)).answer)

    stub_model(env, monkeypatch, message=message)
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert receipts[0]["id"] == receipts[1]["id"]
    record = env.runtime.reminders.read()[0]
    assert record.result_path == result.answer["path"] and record.session_id == "session-1"
    assert record.expires_at == env.now[0] + timedelta(minutes=15)
    assert len(env.runtime.reminders.read()) == 1 and env.runtime.tasks.messages == {}
    with pytest.raises(ValueError, match="live message query"):
        await send(env, bindings[0])


@pytest.mark.asyncio
async def test_history_race_requires_reconsideration_before_cron_send(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])

    async def message(binding):
        await history(env, binding)
        temporary = await env.runtime.temp_tasks.create("明确约定的提醒")
        await env.runtime.temp_tasks.patch(temporary.id, temporary.revision, status="running")
        await env.runtime.temp_tasks.send(temporary.id, temporary.revision, "另一个 Agent 的新提醒")
        with pytest.raises(ValueError, match="history changed"):
            await send(env, binding)
        assert len(env.runtime.reminders.read()) == 1
        refreshed = await history(env, binding)
        assert refreshed.answer[0]["content"] == "另一个 Agent 的新提醒"
        await send(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    assert len(env.runtime.reminders.read()) == 2


@pytest.mark.asyncio
async def test_cancelled_task_cannot_send_and_message_lease_is_revoked(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])

    async def message(binding):
        await history(env, binding)
        configure(env, [])
        with pytest.raises(ValueError, match="changed or was cancelled"):
            await send(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    assert env.runtime.reminders.read() == [] and env.runtime.tasks.messages == {}


@pytest.mark.asyncio
async def test_recovery_after_receipt_does_not_repeat_message_query(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])

    async def message(binding):
        await history(env, binding)
        await send(env, binding)
        raise RuntimeError("model connection lost after accepted send")

    stub_model(env, monkeypatch, message=message)
    with pytest.raises(RuntimeError, match="after accepted"):
        await CronAgentStep(app_context=env.context)(task=task())
    identifier = env.runtime.reminders.read()[0].id
    stub_model(
        env,
        monkeypatch,
        message=AsyncMock(side_effect=AssertionError("must not resend")),
    )
    await CronAgentStep(app_context=env.context)(task=task())
    env.agent.reply.assert_awaited_once()
    assert [record.id for record in env.runtime.reminders.read()] == [identifier]


@pytest.mark.asyncio
async def test_cron_immediate_bypasses_pause_hours_and_keeps_original_session(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])

    async def message(binding):
        env.now[0] = env.now[0].replace(hour=23, minute=30)
        config = await env.runtime.get_proactive_config()
        env.runtime._write_json(
            env.runtime.proactive_config_path,
            {**config.model_dump(mode="json"), "enabled": False},
        )
        env.runtime.session_id = "session-2"
        await history(env, binding)
        result = await send(env, binding)
        assert result.answer["delivery"] == "immediate" and result.answer["announced_at"]

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    assert env.runtime.reminders.read()[0].session_id == "session-1"


@pytest.mark.asyncio
async def test_expired_cron_urgency_is_not_delivered_later(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])

    async def message(binding):
        await history(env, binding)
        await send(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    env.now[0] += timedelta(minutes=16)
    await env.runtime.deliver_pending_reminder()
    record = env.runtime.reminders.read()[0]
    assert (record.status, record.reason) == ("discarded", "expired")
    env.agent.append_session.assert_not_awaited()
    assert env.runtime.tasks.paths("one")


@pytest.mark.asyncio
async def test_no_main_still_publishes_without_notification_query(pa_workspace, monkeypatch):
    env = pa_workspace
    env.runtime.session_id = None
    stub_model(env, monkeypatch)
    result = await CronAgentStep(app_context=env.context)(task=task())
    assert result.answer["status"] == "succeeded" and env.runtime.tasks.paths("one")
    assert env.agent.reply.await_count == 3 and env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_history_pages_cannot_mix_different_snapshots(pa_workspace):
    env = pa_workspace
    async with env.runtime.proactive_round() as current:
        await env.runtime.get_chat_messages(round_id=current.id)
        temporary = await env.runtime.temp_tasks.create("约定提醒")
        await env.runtime.temp_tasks.patch(temporary.id, temporary.revision, status="running")
        await env.runtime.temp_tasks.send(temporary.id, temporary.revision, "分页期间的新消息")
        with pytest.raises(ValueError, match="offset=0"):
            await env.runtime.get_chat_messages(offset=1, round_id=current.id)
        refreshed = await env.runtime.get_chat_messages(round_id=current.id)
        assert refreshed[0]["content"] == "分页期间的新消息"


@pytest.mark.asyncio
async def test_cron_send_cannot_escape_file_permissions(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])

    async def message(binding):
        await history(env, binding)
        with pytest.raises(PermissionError):
            await SendChatMessageImmediateStep(app_context=env.context)(
                **{**binding, "_allowed_paths": []},
                content="紧急信息",
            )

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_message_tools_inherit_the_cron_invocations_file_scope(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    allowed = [str(env.runtime.tasks.task_path("one")), str(env.runtime.cron_runs_path)]

    async def message(binding):
        assert binding["_allowed_paths"] == allowed
        with pytest.raises(PermissionError):
            await history(env, binding)
        with pytest.raises(PermissionError):
            await send(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task(), _allowed_paths=allowed)
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_published_message_phase_recovers_outside_active_hours(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    stub_model(env, monkeypatch, fail_at=3)
    with pytest.raises(RuntimeError, match="provider failed"):
        await CronAgentStep(app_context=env.context)(task=task())
    original = env.runtime.tasks.paths("one")[0].read_bytes()
    env.now[0] = env.now[0].replace(hour=23, minute=30)

    async def message(binding):
        await history(env, binding)
        await send(env, binding)

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    assert env.agent.reply.await_count == 2
    assert env.runtime.tasks.paths("one")[0].read_bytes() == original
    assert len(env.runtime.reminders.read()) == 1


@pytest.mark.asyncio
async def test_revoked_message_query_cannot_commit_a_queued_send(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    submit = env.runtime.reminders.submit_immediate
    entered = asyncio.Event()

    async def queued(*args, **kwargs):
        entered.set()
        return await submit(*args, **kwargs)

    monkeypatch.setattr(env.runtime.reminders, "submit_immediate", queued)

    async def message(binding):
        await history(env, binding)
        async with env.runtime.reminders.lock:
            pending = asyncio.create_task(send(env, binding))
            await asyncio.wait_for(entered.wait(), timeout=3)
            env.runtime.tasks.messages.pop(binding["cron_run_id"])
        with pytest.raises(ValueError, match="live message query"):
            await pending

    stub_model(env, monkeypatch, message=message)
    await CronAgentStep(app_context=env.context)(task=task())
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_stable_native_tools_remain_guarded_outside_message_phase(pa_workspace, monkeypatch):
    env = pa_workspace
    configure(env, [task()])
    stub_model(env, monkeypatch)
    reply = env.agent.reply.side_effect

    async def check(query, **kwargs):
        binding = kwargs["injected_job_kwargs"]
        if binding["cron_run_id"] not in env.runtime.tasks.messages:
            with pytest.raises(ValueError, match="live message query"):
                await history(env, binding)
            with pytest.raises(ValueError, match="live message query"):
                await send(env, binding)
        return await reply(query, **kwargs)

    env.agent.reply = AsyncMock(side_effect=check)
    await CronAgentStep(app_context=env.context)(task=task())
    assert env.agent.reply.await_count == 4 and env.runtime.reminders.read() == []
