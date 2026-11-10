"""Unified one-shot task actions and immediate presentation without idle delays."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import frontmatter

from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.steps.proactive import TempAgentStep, TempTaskLoopStep

from reme.steps.proactive.tools import (
    ManageTempProactiveAgentStep,
    SendChatMessageImmediateStep,
    SendChatMessageStep,
)

from .pa_helpers import wait_until


@pytest_asyncio.fixture(name="temp_agent_job")
async def _temp_agent_job(pa_workspace):
    """Exercise the same private execution job used by the supervisor."""
    env = pa_workspace
    definition = resolve_app_config(config="proactive", log_config=False)["jobs"]["temp_agent"]
    job = BaseJob(name="temp_agent", app_context=env.context, **definition)
    env.context.jobs["temp_agent"] = job
    async with job:
        yield job


@pytest.mark.asyncio
async def test_supervisor_waits_until_due_and_dispatches_once(pa_workspace, temp_agent_job):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("喝水", delay_minutes=5)
    env.agent.reply = AsyncMock(return_value={"status": "completed", "session_id": "fork", "result": "喝水时间到。"})
    stop = asyncio.Event()
    supervisor = asyncio.create_task(TempTaskLoopStep(app_context=env.context)(stop_event=stop))
    try:
        await asyncio.sleep(0.02)
        env.agent.reply.assert_not_awaited()
        env.agent.create_session.assert_not_awaited()
        env.now[0] = task.run_at
        await wait_until(lambda: env.runtime.temp_tasks.get(task.id).status == "completed")
        env.agent.reply.assert_awaited_once()
        env.agent.create_session.assert_not_awaited()
        assert env.context.get_job_count(temp_agent_job.name) == 1
        assert len(env.runtime.reminders.read()) == 1
    finally:
        stop.set()
        await supervisor


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_supervisor_awaits_cancelled_workers_without_publishing(pa_workspace, temp_agent_job, shutdown):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("调查论文")
    env.runtime.event_stream.start()
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def reply(*_args, **_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    env.agent.reply = AsyncMock(side_effect=reply)
    stop = asyncio.Event()
    supervisor = asyncio.create_task(TempTaskLoopStep(app_context=env.context)(stop_event=stop))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert f"temp_{task.id}" in env.runtime.event_stream._task_activities
        if shutdown:
            stop.set()
        else:
            await env.runtime.temp_tasks.manage(task.id, "cancel")
        await asyncio.wait_for(cancelled.wait(), 3)
    finally:
        stop.set()
        await supervisor
    assert env.runtime.reminders.read() == []
    assert not env.runtime.temp_tasks.document_path(task.id).exists()
    assert env.runtime.event_stream._task_activities == {}
    assert env.context.job_history[temp_agent_job.name][-1].status == "cancelled"
    await env.runtime.event_stream.close()


@pytest.mark.asyncio
async def test_temp_job_rejects_early_and_stale_work_and_serializes_duplicates(pa_workspace, temp_agent_job):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("调查论文", delay_minutes=5)
    env.agent.reply = AsyncMock(return_value={"status": "completed", "session_id": "fork", "result": "结果。"})
    early = await temp_agent_job(task_id=task.id, revision=task.revision)
    assert not early.success and "not due" in early.answer
    env.agent.reply.assert_not_awaited()
    env.now[0] = task.run_at
    results = await asyncio.gather(
        temp_agent_job(task_id=task.id, revision=task.revision),
        temp_agent_job(task_id=task.id, revision=task.revision),
        return_exceptions=True,
    )
    assert sum(isinstance(result, asyncio.CancelledError) for result in results) == 1
    assert next(result for result in results if not isinstance(result, BaseException)).success
    env.agent.reply.assert_awaited_once()
    assert len(env.runtime.reminders.read()) == 1
    pending = await env.runtime.temp_tasks.create("尚未执行")
    revised = await env.runtime.temp_tasks.manage(pending.id, "update", query="修订后执行")
    with pytest.raises(asyncio.CancelledError):
        await temp_agent_job(task_id=pending.id, revision=pending.revision)
    assert env.runtime.temp_tasks.get(pending.id).revision == revised.revision
    env.agent.reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_registered_tool_creates_retries_and_manages_one_task(pa_workspace):
    env = pa_workspace
    config = resolve_app_config(config="proactive", log_config=False)["jobs"]
    job = BaseJob(name="manage_temp_proactive_agent", app_context=env.context, **config["manage_temp_proactive_agent"])
    async with job:
        options = {"action": "create", "query": "调查机器人新论文", "delay_minutes": 10, "request_id": "same-request"}
        first = await job(pa_caller="chat", **options)
        retry = await job(pa_caller="chat", **{**options, "delay_minutes": 10.0})
        assert first.success and retry.success and first.answer == retry.answer
        identifier = first.answer["id"]
        assert first.answer["source_session_id"] == "session-1"
        assert first.answer["run_at"] == "2026-11-08T12:10:00+08:00"
        for action, expected in [
            ("update", "scheduled"),
            ("pause", "paused"),
            ("resume", "scheduled"),
            ("cancel", "cancelled"),
        ]:
            changes = {"query": "只看开源项目"} if action == "update" else {}
            result = await job(pa_caller="chat", action=action, task_id=identifier, **changes)
            assert result.success and result.answer["id"] == identifier and result.answer["status"] == expected
        assert len(env.runtime.temp_tasks.load()) == 1
        assert env.runtime.temp_tasks.get(identifier).revision == 4
        with pytest.raises(asyncio.CancelledError):
            await env.runtime.temp_tasks.patch(identifier, 0, status="sending")


@pytest.mark.asyncio
async def test_registered_tool_lists_task_details_without_writing_state(pa_workspace):
    env = pa_workspace
    definition = resolve_app_config(config="proactive", log_config=False)["jobs"]["manage_temp_proactive_agent"]
    assert "list" in definition["parameters"]["properties"]["action"]["enum"]
    job = BaseJob(name="manage_temp_proactive_agent", app_context=env.context, **definition)
    async with job:
        empty = await job(pa_caller="chat", action="list")
        assert empty.success and empty.answer == {"items": [], "total": 0}
        first = await job(pa_caller="chat", action="create", query="提醒看论文", delay_minutes=5.0)
        second = await job(pa_caller="chat", action="create", query="提醒看美股", delay_minutes=10.0)
        paused = await job(pa_caller="chat", action="pause", task_id=first.answer["id"])
        cancelled = await job(pa_caller="chat", action="cancel", task_id=second.answer["id"])
        assert paused.success and cancelled.success
        before = env.runtime.temp_tasks.path.read_bytes()
        page = await job(
            pa_caller="chat",
            action="list",
            _allowed_paths=[str(env.runtime.temp_tasks.path)],
        )
    assert page.success and page.answer["total"] == 2
    items = {item["id"]: item for item in page.answer["items"]}
    assert items[first.answer["id"]] == paused.answer
    assert items[second.answer["id"]] == cancelled.answer
    assert items[first.answer["id"]]["run_at"] == "2026-11-08T12:05:00+08:00"
    assert env.runtime.temp_tasks.path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [0.0, 0.5, 1.25])
async def test_fractional_minutes_are_preserved_when_creating_and_updating(pa_workspace, delay):
    env = pa_workspace
    tool = ManageTempProactiveAgentStep(app_context=env.context)
    created = await tool(pa_caller="chat", action="create", query="看一下论文", delay_minutes=delay)
    assert created.answer["run_at"] == (env.now[0] + timedelta(minutes=delay)).isoformat()
    updated = await tool(
        pa_caller="chat",
        action="update",
        task_id=created.answer["id"],
        delay_minutes=delay + 0.25,
    )
    expected = env.now[0] + timedelta(minutes=delay + 0.25)
    assert updated.answer["run_at"] == expected.isoformat()
    assert env.runtime.temp_tasks.get(created.answer["id"]).run_at == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [-0.5, True, "0.5", float("nan"), float("inf")])
async def test_invalid_delay_does_not_create_a_task(pa_workspace, delay):
    env = pa_workspace
    before = env.runtime.temp_tasks.path.read_bytes()
    with pytest.raises(ValueError, match="non-negative finite number"):
        await ManageTempProactiveAgentStep(app_context=env.context)(
            pa_caller="chat",
            action="create",
            query="提醒我",
            delay_minutes=delay,
        )
    assert env.runtime.temp_tasks.path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,match",
    [
        ({"action": "create"}, "query"),
        ({"action": "create", "query": "work", "task_id": "existing"}, "task_id"),
        ({"action": "create", "query": "work", "run_at": "2026-11-08T13:00:00+08:00", "delay_minutes": 1}, "either"),
        ({"action": "create", "query": "work", "run_at": "2026-11-08T13:00:00"}, "timezone"),
        ({"action": "update", "query": "work"}, "task_id"),
        ({"action": "pause", "task_id": "existing", "request_id": "key"}, "Only create"),
        ({"action": "unknown"}, "action must"),
        ({"action": "list", "task_id": "existing"}, "list does not accept"),
        ({"action": "list", "query": "work"}, "list does not accept"),
        ({"action": "list", "run_at": "2026-11-08T13:00:00+08:00"}, "list does not accept"),
        ({"action": "list", "delay_minutes": 0.5}, "list does not accept"),
        ({"action": "list", "request_id": "key"}, "list does not accept"),
    ],
)
async def test_unified_tool_rejects_invalid_action_parameters(pa_workspace, options, match):
    env = pa_workspace
    with pytest.raises(ValueError, match=match):
        await ManageTempProactiveAgentStep(app_context=env.context)(pa_caller="chat", **options)
    assert env.runtime.temp_tasks.load() == []


@pytest.mark.asyncio
async def test_existing_task_actions_reject_unrelated_fields(pa_workspace):
    env = pa_workspace
    tool = ManageTempProactiveAgentStep(app_context=env.context)
    created = await tool(pa_caller="chat", action="create", query="work")
    for action, changes in [("pause", {"query": "changed"}), ("resume", {"delay_minutes": 1}), ("update", {})]:
        with pytest.raises(ValueError):
            await tool(pa_caller="chat", action=action, task_id=created.answer["id"], **changes)
    assert env.runtime.temp_tasks.get(created.answer["id"]).revision == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("caller,scope", [("proactive", None), ("temporary", None), ("chat", [])])
@pytest.mark.parametrize("action", ["create", "list"])
async def test_unified_tool_preserves_chat_and_path_permissions(pa_workspace, caller, scope, action):
    options = {"query": "work"} if action == "create" else {}
    with pytest.raises((ValueError, PermissionError)):
        await ManageTempProactiveAgentStep(app_context=pa_workspace.context)(
            pa_caller=caller,
            _allowed_paths=scope,
            action=action,
            **options,
        )
    assert pa_workspace.runtime.temp_tasks.load() == []


@pytest.mark.asyncio
async def test_temporary_result_announces_while_chat_busy_then_inserts_without_ten_second_delay(pa_workspace):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("调查新进展")
    task = await env.runtime.temp_tasks.patch(task.id, task.revision, status="running")
    options = {
        "pa_caller": "temporary",
        "temp_task_id": task.id,
        "temp_revision": task.revision,
        "content": "调查结果。",
    }
    async with env.runtime.session_lock:
        config = resolve_app_config(config="proactive", log_config=False)["jobs"]
        job = BaseJob(
            name="send_chat_message_immediate",
            app_context=env.context,
            **config["send_chat_message_immediate"],
        )
        async with job:
            result = await job(**options)
        assert result.success
        assert result.answer["delivery"] == "immediate"
        record = env.runtime.reminders.read()[0]
        assert record.announced_at is not None and record.status == "pending"
        await env.runtime.deliver_pending_reminder()
        env.agent.append_session.assert_not_awaited()
    await env.runtime.deliver_pending_reminder()
    env.agent.append_session.assert_awaited_once()
    assert env.agent.append_session.call_args.kwargs["delivery"] == "immediate"
    assert env.runtime.reminders.read()[0].status == "sent" and env.runtime.reminders.idle is None
    await SendChatMessageImmediateStep(app_context=env.context)(**options)
    assert len(env.runtime.reminders.read()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("delayed", [False, True])
async def test_temporary_worker_forks_chat_and_publishes_final_result(pa_workspace, language, delayed):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create(
        "主人关注本地 Agent。调研其长期记忆能力，简短分点并保留来源。",
        run_at=env.now[0] - timedelta(seconds=10 if delayed else 0),
    )
    env.agent.reply = AsyncMock(
        return_value={
            "session_id": "fork-worker",
            "status": "completed",
            "result": "调查结果。",
        },
    )
    await TempAgentStep(app_context=env.context, language=language)._run(task)
    saved = env.runtime.temp_tasks.get(task.id)
    assert saved.status == "completed" and saved.error is None and saved.delayed is delayed
    env.agent.reply.assert_awaited_once()
    env.agent.create_session.assert_not_awaited()
    execution = env.agent.reply.call_args.kwargs
    assert execution["resume"] == task.source_session_id and execution["fork_session"] is True
    assert not {"model", "job_tools", "system_prompt"} & execution.keys()
    assert execution["injected_job_kwargs"] == {"pa_caller": "temporary"}
    assert (
        task.query in env.agent.reply.call_args.args[0] and env.now[0].isoformat() in env.agent.reply.call_args.args[0]
    )
    assert saved.agent_session_id == "fork-worker"
    assert frontmatter.load(env.runtime.workspace_path / saved.result_path).content == "调查结果。"
    assert saved.message_id == env.runtime.reminders.read()[0].id
    assert env.runtime.reminders.read()[0].content.startswith("【已延迟】") is delayed


@pytest.mark.asyncio
async def test_interrupted_query_is_not_automatically_reissued_on_cold_start(pa_workspace):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("work")
    task = await env.runtime.temp_tasks.patch(task.id, task.revision, queried=True, status="running")
    env.agent.reply = AsyncMock()
    await TempAgentStep(app_context=env.context)._run(task)
    saved = env.runtime.temp_tasks.get(task.id)
    assert saved.status == "failed" and "interrupted" in saved.error
    env.agent.reply.assert_not_awaited()
    env.agent.create_session.assert_not_awaited()
    resumed = await env.runtime.temp_tasks.manage(task.id, "resume")
    assert resumed.revision == 1 and not resumed.queried and resumed.status == "scheduled"


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", [None, "chat", "proactive"])
async def test_immediate_tool_requires_temporary_binding(pa_workspace, caller):
    env = pa_workspace
    with pytest.raises(ValueError, match="server-bound task"):
        await SendChatMessageImmediateStep(app_context=env.context)(pa_caller=caller, content="result")
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_immediate_tool_rejects_wrong_state_revision_and_permissions(pa_workspace):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("work")
    options = {"pa_caller": "temporary", "temp_task_id": task.id, "temp_revision": task.revision, "content": "result"}
    tool = SendChatMessageImmediateStep(app_context=env.context)
    with pytest.raises(ValueError, match="running"):
        await tool(**options)
    await env.runtime.temp_tasks.patch(task.id, task.revision, status="running")
    with pytest.raises(PermissionError):
        await tool(**options, _allowed_paths=[])
    with pytest.raises(ValueError, match="active"):
        await tool(**{**options, "temp_revision": task.revision + 1})
    with pytest.raises(ValueError, match="server-bound sending"):
        await SendChatMessageStep(app_context=env.context)(**options)
    assert not env.runtime.temp_tasks.document_path(task.id).exists()
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   ", None, 42])
async def test_immediate_tool_does_not_checkpoint_invalid_results(pa_workspace, content):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("work")
    await env.runtime.temp_tasks.patch(task.id, task.revision, status="running")
    with pytest.raises(ValueError, match="content must"):
        await env.runtime.temp_tasks.send(task.id, task.revision, content)
    assert not env.runtime.temp_tasks.document_path(task.id).exists()
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_temporary_worker_recovers_checkpointed_result_without_model_call(pa_workspace, monkeypatch):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("work")
    task = await env.runtime.temp_tasks.patch(task.id, task.revision, status="running", agent_session_id="worker")
    submit = env.runtime.reminders.submit_immediate
    monkeypatch.setattr(env.runtime.reminders, "submit_immediate", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await env.runtime.temp_tasks.send(task.id, task.revision, "saved result")
    saved = env.runtime.temp_tasks.get(task.id)
    assert saved.status == "sending" and saved.result_path
    monkeypatch.setattr(env.runtime.reminders, "submit_immediate", submit)
    env.agent.reply = AsyncMock(side_effect=AssertionError("Recovery must not reexecute the query"))

    await TempAgentStep(app_context=env.context)._run(saved)

    assert env.runtime.temp_tasks.get(task.id).status == "completed"
    assert env.runtime.reminders.read()[0].content == "saved result"
    env.agent.reply.assert_not_awaited()
