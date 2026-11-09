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
        await env.runtime.get_chat_messages(round_id=current.id)
        return await env.runtime.send_chat_message(
            content,
            kind="normal",
            round_id=current.id,
            session_id=current.session_id,
        )


@pytest.mark.asyncio
async def test_reminder_preserves_context_and_sources_beyond_the_previous_brief_limit(pa_workspace):
    env = pa_workspace
    content = (
        "核心结论：当前资料只能确认测试范围内的行为。\n"
        + "必要背景：本轮验证覆盖读取、筛选和恢复；浏览器界面及真实服务仍待验证。\n" * 8
        + "[验证报告](https://example.test/reports/validation)"
    )
    assert len(content) > 200
    result = await submit(env, content)
    assert result["status"] == "pending"
    assert env.runtime.reminders.read()[0].content == content
    assert (await env.runtime.get_chat_messages())[0]["content"] == content


@pytest.mark.asyncio
async def test_reminder_size_boundary_matches_the_tool_schema_without_consuming_a_rejected_decision(pa_workspace):
    from reme.config import resolve_app_config

    env = pa_workspace
    jobs = resolve_app_config(config="proactive", log_config=False)["jobs"]
    limit = jobs["send_chat_message"]["parameters"]["properties"]["content"]["maxLength"]
    async with env.runtime.proactive_round() as current:
        current.phase = "query_message"
        await env.runtime.get_chat_messages(round_id=current.id)
        options = {"kind": "normal", "round_id": current.id, "session_id": current.session_id}
        with pytest.raises(ValueError, match="at most"):
            await env.runtime.send_chat_message("文" * (limit + 1), **options)
        assert current.request is None and env.runtime.reminders.read() == []
        result = await env.runtime.send_chat_message("文" * limit, **options)
        assert result["status"] == "pending" and env.runtime.reminders.read()[0].content == "文" * limit


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
        await env.runtime.get_chat_messages(round_id=current.id)
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
        await env.runtime.get_chat_messages(round_id=current.id)

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
        await env.runtime.get_chat_messages(round_id=current.id)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ordinary", "immediate"])
async def test_transport_failure_survives_restart_and_retries_same_message(pa_workspace, mode):
    env = pa_workspace
    if mode == "ordinary":
        await submit(env, "可重试的提醒")
        env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    else:
        task = await env.runtime.temp_tasks.create("约定任务")
        await env.runtime.temp_tasks.patch(task.id, task.revision, status="running")
        await env.runtime.temp_tasks.send(task.id, task.revision, "可重试的提醒")
    identifier = env.runtime.reminders.read()[0].id
    env.agent.append_session = AsyncMock(side_effect=ConnectionError("temporary transport failure"))
    await env.runtime.deliver_pending_reminder()
    record = env.runtime.reminders.read()[0]
    assert record.status == "pending" and record.retry_at and record.error
    env.agent.append_session.side_effect = None
    env.runtime.reminders.restore()
    await env.runtime.deliver_pending_reminder()
    assert env.agent.append_session.await_count == 1
    env.now[0] += timedelta(seconds=6)
    env.runtime.reminders.idle = ("turn-1", asyncio.get_running_loop().time() - 11)
    await env.runtime.deliver_pending_reminder()
    assert env.agent.append_session.await_count == 2
    assert env.runtime.reminders.read()[0].id == identifier
    assert env.runtime.reminders.read()[0].status == "sent"
    assert env.agent.append_session.await_args.kwargs["message_id"] == identifier


@pytest.mark.asyncio
async def test_history_change_rejects_stale_proactive_decision(pa_workspace):
    env = pa_workspace
    async with env.runtime.proactive_round() as current:
        current.phase = "query_message"
        await env.runtime.get_chat_messages(round_id=current.id)
        task = await env.runtime.temp_tasks.create("约定提醒")
        await env.runtime.temp_tasks.patch(task.id, task.revision, status="running")
        await env.runtime.temp_tasks.send(task.id, task.revision, "新的即时提醒")
        with pytest.raises(ValueError, match="history changed"):
            await env.runtime.send_chat_message(
                "旧快照决策",
                "normal",
                round_id=current.id,
                session_id=current.session_id,
            )
        assert current.request is None
        await env.runtime.get_chat_messages(round_id=current.id)
        result = await env.runtime.send_chat_message(
            "重新判断后的信息",
            "normal",
            round_id=current.id,
            session_id=current.session_id,
        )
        assert result["status"] == "pending"


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["runtime", "tool"])
async def test_every_sending_entry_point_requires_history_without_consuming_the_decision(pa_workspace, caller):
    env = pa_workspace
    async with env.runtime.proactive_round() as current:
        current.phase = "query_message"

        async def send():
            if caller == "tool":
                response = await SendChatMessageStep(app_context=env.context)(
                    content="新的发现。",
                    kind="normal",
                    pa_caller="proactive",
                    pa_round_id=current.id,
                    session_id=current.session_id,
                )
                return response.answer
            return await env.runtime.send_chat_message(
                "新的发现。",
                "normal",
                round_id=current.id,
                session_id=current.session_id,
            )

        with pytest.raises(ValueError, match="get_chat_messages"):
            await send()
        assert current.request is None and env.runtime.reminders.read() == []
        await env.runtime.get_chat_messages(round_id=current.id)
        assert (await send())["status"] == "pending"


@pytest.mark.asyncio
async def test_external_journal_edit_during_history_read_requires_a_fresh_snapshot(pa_workspace, monkeypatch):
    from reme.schema.reminder import ReminderRecord

    env = pa_workspace
    read = env.runtime.reminders.read
    record = ReminderRecord(
        content="另一个写入者的新消息。",
        kind="normal",
        status="sent",
        session_id="session-1",
        created_at=env.now[0],
        updated_at=env.now[0],
        expires_at=env.now[0] + timedelta(minutes=5),
    )

    def changed():
        records = read()
        env.runtime._append_jsonl(env.runtime.chat_inbox_path, record.model_dump(mode="json"))
        return records

    async with env.runtime.proactive_round() as current:
        monkeypatch.setattr(env.runtime.reminders, "read", changed)
        with pytest.raises(ValueError, match="history changed"):
            await env.runtime.get_chat_messages(round_id=current.id)
        assert current.history_version is None
        monkeypatch.setattr(env.runtime.reminders, "read", read)
        rows = await env.runtime.get_chat_messages(round_id=current.id)
        assert rows[0]["content"] == record.content


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["closed", "busy", "invalid"])
@pytest.mark.parametrize("operation", ["read_session", "get_session_status", "append_session"])
async def test_codex_sdk_transient_session_errors_use_the_retry_contract(pa_workspace, monkeypatch, kind, operation):
    from reme.components.agent_wrapper import CodexAgentWrapper

    errors = pytest.importorskip("openai_codex.errors")
    error = (
        errors.TransportClosedError("closed")
        if kind == "closed"
        else (
            errors.ServerBusyError(-32000, "busy") if kind == "busy" else errors.InvalidParamsError(-32602, "invalid")
        )
    )
    env = pa_workspace
    monkeypatch.setattr(env.agent, "start", AsyncMock())
    monkeypatch.setattr(env.agent.app_server, operation, AsyncMock(side_effect=error))
    args = ("session-1", "result") if operation == "append_session" else ("session-1",)
    kwargs = {"message_id": "stable-id"} if operation == "append_session" else {}
    with pytest.raises(type(error) if kind == "invalid" else ConnectionError):
        await getattr(CodexAgentWrapper, operation)(env.agent, *args, **kwargs)
