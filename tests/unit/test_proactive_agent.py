"""Seven-query orchestration, exclusive rounds and durable interrupted-input recovery."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from unittest.mock import AsyncMock

import pytest

from reme.schema.pa_proactive import PaProactiveConfig
from reme.steps.proactive import ProactiveAgentStep, TaskAgentStep
from reme.steps.proactive.tools import GetProactiveMessagesStep, SendReminderStep


def model(env, monkeypatch, *, fail_at=None, decide=True, adjust=False, pause=None):
    calls = []
    monkeypatch.setattr(ProactiveAgentStep, "_now", lambda _step: env.now[0])

    async def reply(query, **kwargs):
        calls.append((query, kwargs))
        stage = (len(calls) - 1) % 7 + 1
        bound = kwargs["injected_job_kwargs"]
        if stage == 2:
            await env.runtime.get_proactive_messages(round_id=bound["pa_round_id"])
            if pause is not None:
                pause[0].set()
                await pause[1].wait()
        if stage == fail_at:
            raise RuntimeError("query failed")
        if stage == 5 and decide:
            await env.runtime.send_reminder(
                "",
                kind="normal",
                round_id=bound["pa_round_id"],
                session_id=bound["session_id"],
            )
        if stage == 6 and adjust:
            env.runtime._write_json(
                env.runtime.proactive_config_path,
                {"start_time": "0900", "end_time": "2300", "interval_minutes": 15},
            )
        return {
            "session_id": kwargs.get("resume") or f"agent-{bound['pa_round_id']}",
            "status": "completed",
            "result": f"stage {stage}",
        }

    env.agent.reply = AsyncMock(side_effect=reply)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
async def test_seven_queries_share_session_and_tools_then_save_handoff(pa_workspace, monkeypatch, language):
    env = pa_workspace
    calls = model(env, monkeypatch, adjust=True)
    step = ProactiveAgentStep(app_context=env.context, language=language, job_tools=["send_reminder", "check_json"])
    result = await step()
    assert len(calls) == 7 and calls[0][1]["resume"] is None
    assert calls[3][0] == step.get_prompt("query_research")
    assert len({kwargs["resume"] for _, kwargs in calls[1:]}) == 1
    assert len({str(kwargs["injected_job_kwargs"]) for _, kwargs in calls}) == 1
    assert all(kwargs["job_tools"] == ["send_reminder", "check_json"] for _, kwargs in calls)
    assert "15" in calls[-1][0] and "12:15" in calls[-1][0]
    assert result.answer["handoff"] == "stage 7" and env.runtime.handoff_path.read_text().strip() == "stage 7"
    assert env.runtime.proactive_context_path.read_text() == "" and env.runtime.current_round is None


@pytest.mark.asyncio
async def test_only_reminder_query_can_submit_despite_an_old_handoff(pa_workspace, monkeypatch):
    env = pa_workspace
    await env.runtime.write_handoff("旧交接：必须立即调用 send_reminder 推送，不要重新判断。")
    monkeypatch.setattr(ProactiveAgentStep, "_now", lambda _step: env.now[0])
    phases = []
    rounds = []

    async def reply(_query, **kwargs):
        current = env.runtime.current_round
        phases.append(current.phase)
        rounds.append(current)
        args = {**kwargs["injected_job_kwargs"], "content": "", "kind": "normal"}
        tool = SendReminderStep(app_context=env.context)
        if current.phase == "query_reminder":
            await tool(**args)
        else:
            with pytest.raises(ValueError, match="only allowed during query_reminder"):
                await tool(**args)
        return {"session_id": "agent", "status": "completed", "result": "saved facts"}

    env.agent.reply = AsyncMock(side_effect=reply)
    result = await ProactiveAgentStep(app_context=env.context)()
    assert phases == [
        "query_memory",
        "query_conversation",
        "query_tasks",
        "query_research",
        "query_reminder",
        "query_config",
        "query_handoff",
    ]
    assert result.answer["status"] == "skipped" and all(current.phase is None for current in rounds)
    assert await env.runtime.get_reminder() == []
    env.agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_failure_keeps_previous_handoff_and_consumed_feedback(pa_workspace, monkeypatch):
    env = pa_workspace
    await env.runtime.write_handoff("previous summary")
    await env.runtime.send_to_proactive("只跟踪主人关注的新结果")
    model(env, monkeypatch, fail_at=3)
    with pytest.raises(RuntimeError, match="query failed"):
        await ProactiveAgentStep(app_context=env.context)()
    assert env.runtime.handoff_path.read_text().strip() == "previous summary"
    assert env.runtime.proactive_inbox_path.read_text() == ""
    recovery = env.runtime.proactive_context_path.read_text()
    assert "只跟踪主人关注的新结果" in recovery and "failure" in recovery
    assert env.runtime.current_round is None and not env.runtime._proactive_lock.locked()


@pytest.mark.asyncio
async def test_failed_research_preserves_recovery_without_submitting_a_reminder(pa_workspace, monkeypatch):
    env = pa_workspace
    await env.runtime.write_handoff("previous summary")
    calls = model(env, monkeypatch, fail_at=4)
    send = AsyncMock(wraps=env.runtime.send_reminder)
    monkeypatch.setattr(env.runtime, "send_reminder", send)
    with pytest.raises(RuntimeError, match="query failed"):
        await ProactiveAgentStep(app_context=env.context)()
    assert len(calls) == 4
    send.assert_not_awaited()
    assert env.runtime.handoff_path.read_text().strip() == "previous summary"
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
    assert current.phase == "query_conversation"
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
    assert len(calls) == 14
    assert calls[0][1]["injected_job_kwargs"]["pa_round_id"] != calls[7][1]["injected_job_kwargs"]["pa_round_id"]


@pytest.mark.asyncio
async def test_missing_reminder_decision_is_not_reported_as_success(pa_workspace, monkeypatch):
    model(pa_workspace, monkeypatch, decide=False)
    with pytest.raises(RuntimeError, match="submit a send_reminder decision"):
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
    await env.runtime.send_to_proactive("保留反馈")
    original = env.runtime.proactive_inbox_path.read_bytes()
    monkeypatch.setattr(env.runtime, "_checkpoint_context", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        await GetProactiveMessagesStep(app_context=env.context)(pa_round_id="round")
    assert env.runtime.proactive_inbox_path.read_bytes() == original


@pytest.mark.asyncio
async def test_task_agent_requires_a_complete_task_definition(pa_workspace):
    env = pa_workspace
    env.agent.reply = AsyncMock(return_value={"session_id": "task", "status": "completed", "result": "result"})
    with pytest.raises(ValueError):
        await TaskAgentStep(app_context=env.context)(task={"query": "理解本地结果"})
    env.agent.reply.assert_not_awaited()


@pytest.mark.parametrize("minutes", [0, -1, True, "5", float("nan"), float("inf")])
def test_wakeup_interval_is_a_positive_finite_number(minutes):
    with pytest.raises(ValueError):
        PaProactiveConfig(start_time="0900", end_time="2300", interval_minutes=minutes)
