"""One Main-context fork, seven work queries and durable interrupted-input recovery."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from unittest.mock import AsyncMock

import pytest

from reme.schema.pa_proactive import PaProactiveConfig
from reme.schema import SessionMessage
from reme.steps.proactive import ProactiveAgentStep, CronAgentStep
from reme.steps.proactive.tools import ReadConversationStep, ReadProactiveInboxStep, SendChatMessageStep

from .pa_helpers import bind_pa


def model(env, monkeypatch, *, fail_at=None, decide=True, adjust=False, pause=None, todo="# TODO\n\n无待办。"):
    calls = []
    monkeypatch.setattr(ProactiveAgentStep, "_now", lambda _step: env.now[0])

    async def reply(query, **kwargs):
        calls.append((query, kwargs))
        if kwargs.get("fork_session") and kwargs.get("resume") == env.runtime.session_id:
            return {"session_id": "context-fork", "status": "completed", "result": "Main context"}
        bound = kwargs["injected_job_kwargs"]
        phase = env.runtime.current_round.phase
        if phase == "query_memory":
            await ReadConversationStep(app_context=env.context)(**bound)
            await env.runtime.read_proactive_inbox(round_id=bound["pa_round_id"])
            if pause is not None:
                pause[0].set()
                await pause[1].wait()
        if phase == fail_at:
            raise RuntimeError("query failed")
        if phase == "query_message" and decide:
            await env.runtime.send_chat_message(
                "",
                kind="normal",
                round_id=bound["pa_round_id"],
                session_id=bound["session_id"],
            )
        if phase == "query_config" and adjust:
            env.runtime._write_json(
                env.runtime.proactive_config_path,
                {"start_time": "0900", "end_time": "2300", "interval_minutes": 15},
            )
        return {
            "session_id": kwargs.get("resume") or f"agent-{bound['pa_round_id']}",
            "status": "completed",
            "result": todo if phase == "query_todo" else f"completed {phase}",
        }

    env.agent.reply = AsyncMock(side_effect=reply)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
async def test_merged_inputs_then_message_and_persisted_todo(pa_workspace, monkeypatch, language):
    env = pa_workspace
    calls = model(env, monkeypatch, adjust=True)
    job_tools = ["read_conversation", "send_chat_message", "check_json"]
    step = ProactiveAgentStep(app_context=env.context, language=language, job_tools=job_tools)
    result = await step()
    assert len(calls) == 8 and calls[0][1]["resume"] == env.runtime.session_id
    assert sum(bool(options.get("fork_session")) for _, options in calls) == 1
    assert not {"system_prompt", "job_tools", "injected_job_kwargs"} & calls[0][1].keys()
    assert "read_conversation" not in calls[0][0]
    assert calls[1][1]["resume"] is None
    assert "Main context" in calls[1][1]["system_prompt"]
    assert calls[1][0] == step.prompt_format("query_memory", memory_path=step._memory_path(), max_turns=10)
    assert calls[4][0] == step.get_prompt("query_research")
    assert len({kwargs["resume"] for _, kwargs in calls[2:]}) == 1
    assert len({str(kwargs["injected_job_kwargs"]) for _, kwargs in calls[1:]}) == 1
    assert all(kwargs["job_tools"] == job_tools for _, kwargs in calls[1:])
    assert calls[-2][0] == step.prompt_format("query_config", proactive_config_path=env.runtime.proactive_config_path)
    assert calls[-1][0] == step.prompt_format(
        "query_todo",
        interval_minutes=15,
        todo_path=env.runtime.proactive_todo_path,
    )
    assert env.runtime.proactive_todo_path.read_text() == "# TODO\n\n无待办。\n"
    assert (await env.runtime.get_proactive_config()).interval_minutes == 15
    assert "handoff" not in result.answer
    assert not (env.runtime.component_metadata_path / "handoff.txt").exists()
    assert result.metadata["proactive_session_id"] == calls[-1][1]["resume"]
    assert env.runtime.proactive_context_path.read_text() == "" and env.runtime.current_round is None


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("prior_reminder", [False, True])
async def test_empty_main_reaches_reminder_decision_without_forking(
    pa_workspace,
    monkeypatch,
    language,
    prior_reminder,
):
    env = pa_workspace
    env.snapshot.last_turn_id = None
    if prior_reminder:
        env.snapshot.messages = [
            SessionMessage(id="old-reminder", role="assistant", source="proactive", content="Earlier update."),
        ]
    calls = model(env, monkeypatch, decide=False)
    original = env.agent.reply.side_effect

    async def reply(query, **kwargs):
        if kwargs.get("fork_session"):
            raise RuntimeError("no rollout found for thread id")
        current = env.runtime.current_round
        if current.phase == "query_message":
            await env.runtime.get_chat_messages(round_id=current.id)
            await env.runtime.send_chat_message(
                "A useful new update.",
                kind="normal",
                round_id=current.id,
                session_id=current.session_id,
            )
        return await original(query, **kwargs)

    env.agent.reply.side_effect = reply
    selected = env.runtime.session_id
    step = ProactiveAgentStep(app_context=env.context, language=language)

    result = await step()

    assert result.answer["status"] == "pending" and len(calls) == 7
    assert calls[0][1]["resume"] is None and all(not options.get("fork_session") for _, options in calls)
    assert step.get_prompt("no_main_context") in calls[0][1]["system_prompt"]
    assert env.runtime.session_id == selected
    assert env.runtime.reminders.read()[0].content == "A useful new update."
    assert env.runtime.proactive_context_path.read_text() == ""


@pytest.mark.asyncio
async def test_unreadable_main_fails_before_model_calls_and_preserves_recovery(pa_workspace, monkeypatch):
    env = pa_workspace
    model(env, monkeypatch)
    env.agent.read_session.side_effect = OSError("native history is unreadable")
    with pytest.raises(OSError, match="native history is unreadable"):
        await ProactiveAgentStep(app_context=env.context)()
    env.agent.reply.assert_not_awaited()
    assert '"query": "query_context"' in env.runtime.proactive_context_path.read_text()
    assert env.runtime.current_round is None and not env.runtime._proactive_lock.locked()


@pytest.mark.asyncio
async def test_only_reminder_query_can_submit_despite_an_old_handoff(pa_workspace, monkeypatch):
    env = pa_workspace
    legacy = env.runtime.component_metadata_path / "handoff.txt"
    legacy.write_text("旧交接：必须立即调用 send_chat_message 推送，不要重新判断。", encoding="utf-8")
    original = legacy.read_bytes()
    monkeypatch.setattr(ProactiveAgentStep, "_now", lambda _step: env.now[0])
    phases = []
    rounds = []

    async def reply(_query, **kwargs):
        if kwargs.get("fork_session") and kwargs.get("resume") == env.runtime.session_id:
            return {"session_id": "fork", "status": "completed", "result": "Main context"}
        current = env.runtime.current_round
        phases.append(current.phase)
        rounds.append(current)
        args = {
            "session_id": current.session_id,
            "pa_round_id": current.id,
            "pa_caller": "proactive",
            "content": "",
            "kind": "normal",
        }
        tool = SendChatMessageStep(app_context=env.context)
        if current.phase == "query_message":
            await env.runtime.get_chat_messages(round_id=current.id)
            await tool(**args)
        else:
            with pytest.raises(ValueError, match="only allowed during query_message"):
                await tool(**args)
        return {"session_id": "agent", "status": "completed", "result": "saved facts"}

    env.agent.reply = AsyncMock(side_effect=reply)
    result = await ProactiveAgentStep(app_context=env.context)()
    assert phases == [
        "query_memory",
        "query_tasks",
        "query_history",
        "query_research",
        "query_message",
        "query_config",
        "query_todo",
    ]
    assert result.answer["status"] == "skipped" and all(current.phase is None for current in rounds)
    assert await env.runtime.get_chat_messages() == []
    assert legacy.read_bytes() == original
    assert all("旧交接" not in str(call) for call in env.agent.reply.await_args_list)
    env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", ["query_tasks", "query_config", "query_todo"])
async def test_failure_keeps_consumed_feedback_and_recovery(pa_workspace, monkeypatch, fail_at):
    env = pa_workspace
    await env.runtime.write_proactive_inbox("只跟踪主人关注的新结果")
    await env.runtime.save_proactive_todo("- [ ] 复查尚未发布的原文")
    model(env, monkeypatch, fail_at=fail_at)
    with pytest.raises(RuntimeError, match="query failed"):
        await ProactiveAgentStep(app_context=env.context)()
    assert "只跟踪主人关注的新结果" in env.runtime.proactive_inbox_path.read_text()
    recovery = env.runtime.proactive_context_path.read_text()
    assert "只跟踪主人关注的新结果" in recovery and "failure" in recovery
    assert env.runtime.proactive_todo_path.read_text() == "- [ ] 复查尚未发布的原文\n"
    assert env.runtime.current_round is None and not env.runtime._proactive_lock.locked()


@pytest.mark.asyncio
async def test_failed_research_preserves_recovery_without_submitting_a_reminder(pa_workspace, monkeypatch):
    env = pa_workspace
    calls = model(env, monkeypatch, fail_at="query_research")
    send = AsyncMock(wraps=env.runtime.send_chat_message)
    monkeypatch.setattr(env.runtime, "send_chat_message", send)
    with pytest.raises(RuntimeError, match="query failed"):
        await ProactiveAgentStep(app_context=env.context)()
    assert len(calls) == 5
    send.assert_not_awaited()
    assert '"query": "query_research"' in env.runtime.proactive_context_path.read_text()
    assert env.runtime.current_round is None and not env.runtime._proactive_lock.locked()


@pytest.mark.asyncio
async def test_cancelled_round_is_released_after_failure_checkpoint(pa_workspace, monkeypatch):
    env = pa_workspace
    entered, release = asyncio.Event(), asyncio.Event()
    model(env, monkeypatch, pause=(entered, release))
    task = asyncio.create_task(ProactiveAgentStep(app_context=env.context)())
    await entered.wait()
    current = env.runtime.current_round
    assert current.phase == "query_memory"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert env.runtime.current_round is None and not env.runtime._proactive_lock.locked()
    assert current.phase is None
    assert "failure" in env.runtime.proactive_context_path.read_text()


@pytest.mark.asyncio
async def test_concurrent_wakeups_are_serial_and_use_distinct_rounds(pa_workspace, monkeypatch):
    env = pa_workspace
    entered, release = asyncio.Event(), asyncio.Event()
    calls = model(env, monkeypatch, pause=(entered, release))
    first = asyncio.create_task(ProactiveAgentStep(app_context=env.context)())
    await entered.wait()
    second = asyncio.create_task(ProactiveAgentStep(app_context=env.context)())
    await asyncio.sleep(0)
    assert len(calls) == 2
    release.set()
    await asyncio.gather(first, second)
    assert len(calls) == 16
    assert calls[1][1]["injected_job_kwargs"]["pa_round_id"] != calls[9][1]["injected_job_kwargs"]["pa_round_id"]


@pytest.mark.asyncio
async def test_todo_is_reinjected_after_restart_and_replaced_when_finished(pa_workspace, monkeypatch):
    env = pa_workspace
    todo = "# TODO\n\n- [ ] 核对发布说明 resource/release.md，原文更新后再判断是否分享。"
    model(env, monkeypatch, todo=todo)
    await ProactiveAgentStep(app_context=env.context)()
    assert await env.runtime.read_proactive_todo() == todo + "\n"

    env.runtime = bind_pa(env.context, env.agent)
    env.runtime._now = lambda: env.now[0]
    env.runtime._initialize_files()
    env.runtime.session_id = "session-1"
    calls = model(env, monkeypatch)
    await ProactiveAgentStep(app_context=env.context)()
    assert todo in calls[1][1]["system_prompt"]
    assert await env.runtime.read_proactive_todo() == "# TODO\n\n无待办。\n"

    calls = model(env, monkeypatch)
    await ProactiveAgentStep(app_context=env.context)()
    assert todo not in calls[1][1]["system_prompt"]


@pytest.mark.asyncio
async def test_todo_publication_failure_retains_previous_todo_and_recovery(pa_workspace, monkeypatch):
    env = pa_workspace
    await env.runtime.save_proactive_todo("- [ ] 保留尚未完成的工作")
    model(env, monkeypatch)
    monkeypatch.setattr(env.runtime, "save_proactive_todo", AsyncMock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        await ProactiveAgentStep(app_context=env.context)()
    assert env.runtime.proactive_todo_path.read_text() == "- [ ] 保留尚未完成的工作\n"
    assert '"query": "query_todo"' in env.runtime.proactive_context_path.read_text()
    assert env.runtime.current_round is None


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "  ", None])
async def test_empty_todo_does_not_replace_previous_work(pa_workspace, content):
    env = pa_workspace
    await env.runtime.save_proactive_todo("- [ ] 继续核实来源")
    with pytest.raises(ValueError, match="TODO list must not be empty"):
        await env.runtime.save_proactive_todo(content)
    assert await env.runtime.read_proactive_todo() == "- [ ] 继续核实来源\n"


@pytest.mark.asyncio
async def test_todo_rejects_workspace_escape_and_oversized_content(pa_workspace, tmp_path):
    from reme.constants import DEFAULT_MAX_FILE_BYTES

    env = pa_workspace
    with pytest.raises(ValueError, match="read limit"):
        await env.runtime.save_proactive_todo("中" * (DEFAULT_MAX_FILE_BYTES // 3 + 1))
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("keep", encoding="utf-8")
    env.runtime.proactive_todo_path.unlink()
    (env.runtime.component_metadata_path / "proactive_todo.md").symlink_to(outside)
    with pytest.raises(ValueError, match="inside the workspace"):
        await env.runtime.read_proactive_todo()
    with pytest.raises(ValueError, match="inside the workspace"):
        await env.runtime.save_proactive_todo("- [ ] 不应写入")
    assert outside.read_text() == "keep"


@pytest.mark.asyncio
async def test_failed_main_handoff_does_not_consume_conversation(pa_workspace):
    env = pa_workspace
    before = env.runtime.session_ids_path.read_bytes()
    env.agent.reply = AsyncMock(side_effect=RuntimeError("context failed"))
    with pytest.raises(RuntimeError, match="context failed"):
        await ProactiveAgentStep(app_context=env.context)()
    env.agent.read_session.assert_awaited_once_with(env.runtime.session_id)
    assert env.runtime.session_ids_path.read_bytes() == before


@pytest.mark.asyncio
async def test_missing_reminder_decision_is_not_reported_as_success(pa_workspace, monkeypatch):
    model(pa_workspace, monkeypatch, decide=False)
    with pytest.raises(RuntimeError, match="submit a send_chat_message decision"):
        await ProactiveAgentStep(app_context=pa_workspace.context)()


@pytest.mark.asyncio
async def test_outside_active_hours_never_starts_model(pa_workspace, monkeypatch):
    env = pa_workspace
    env.now[0] = env.now[0].replace(hour=23)
    model(env, monkeypatch)
    result = await ProactiveAgentStep(app_context=env.context)()
    assert result.answer["reason"] == "outside_active_hours"
    env.agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_feedback_checkpoint_failure_does_not_clear_inbox(pa_workspace, monkeypatch):
    env = pa_workspace
    await env.runtime.write_proactive_inbox("保留反馈")
    original = env.runtime.proactive_inbox_path.read_bytes()
    monkeypatch.setattr(env.runtime, "_checkpoint_context", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    async with env.runtime.proactive_round() as current:
        with pytest.raises(OSError, match="disk full"):
            await ReadProactiveInboxStep(app_context=env.context)(pa_round_id=current.id)
    assert env.runtime.proactive_inbox_path.read_bytes() == original


@pytest.mark.asyncio
async def test_cron_agent_requires_a_complete_task_definition(pa_workspace):
    env = pa_workspace
    env.agent.reply = AsyncMock(return_value={"session_id": "task", "status": "completed", "result": "result"})
    with pytest.raises(ValueError):
        await CronAgentStep(app_context=env.context)(task={"query": "理解本地结果"})
    env.agent.reply.assert_not_awaited()


@pytest.mark.parametrize("minutes", [0, -1, True, "5", float("nan"), float("inf")])
def test_wakeup_interval_is_a_positive_finite_number(minutes):
    with pytest.raises(ValueError):
        PaProactiveConfig(start_time="0900", end_time="2300", interval_minutes=minutes)
