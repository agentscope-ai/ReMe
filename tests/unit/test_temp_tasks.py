"""Unified one-shot task actions and immediate presentation without idle delays."""

# pylint: disable=missing-function-docstring,protected-access

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
import frontmatter

from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.steps.proactive import TempProactiveStep
from reme.steps.proactive.tools import (
    ManageTempProactiveAgentStep,
    SendChatMessageImmediateStep,
    SendChatMessageStep,
)


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
            ("update", "pending"),
            ("pause", "paused"),
            ("resume", "pending"),
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
async def test_unified_tool_preserves_main_and_path_permissions(pa_workspace, caller, scope, action):
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
async def test_temporary_result_announces_while_main_busy_then_inserts_without_ten_second_delay(pa_workspace):
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
@pytest.mark.parametrize("source", ["none", "missing", "file", "directory"])
async def test_temporary_context_only_exposes_an_existing_native_transcript(pa_workspace, monkeypatch, source):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("Read context only when it exists.")
    path = env.runtime.workspace_path / "native-source.jsonl"
    if source == "file":
        path.write_text("{}\n", encoding="utf-8")
    elif source == "directory":
        path.mkdir()
    lookup = AsyncMock(return_value=None if source == "none" else path)
    monkeypatch.setattr(env.agent.app_server, "get_session_path", lookup)
    step = TempProactiveStep(app_context=env.context)

    result = await step._source_path(task)

    assert result == (str(path) if source == "file" else step.get_prompt("source_unavailable"))
    lookup.assert_awaited_once_with(task.source_session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("delayed", [False, True])
async def test_temporary_worker_prepares_executes_and_sends_result(pa_workspace, monkeypatch, language, delayed):
    env = pa_workspace
    run_at = env.now[0] - timedelta(seconds=10 if delayed else 0)
    task = await env.runtime.temp_tasks.create("调查新进展", run_at=run_at)
    calls = []

    async def reply(query, **kwargs):
        calls.append((query, kwargs))
        if len(calls) == 2:
            await SendChatMessageImmediateStep(app_context=env.context)(
                **kwargs["injected_job_kwargs"],
                content="调查结果。",
            )
        return {
            "session_id": "context-fork" if kwargs.get("fork_session") else "temporary-session",
            "status": "completed",
            "result": "Task context" if kwargs.get("fork_session") else "Task result",
        }

    env.agent.reply = AsyncMock(side_effect=reply)
    step = TempProactiveStep(app_context=env.context, language=language, job_tools=["send_chat_message_immediate"])
    monkeypatch.setattr(step, "_source_path", AsyncMock(return_value="source.jsonl"))

    await step._run(task)

    saved = env.runtime.temp_tasks.get(task.id)
    assert saved.status == "completed" and saved.error is None and saved.delayed is delayed
    assert len(calls) == 2
    assert calls[0][1]["resume"] == task.source_session_id and calls[0][1]["fork_session"]
    assert not {"system_prompt", "job_tools", "injected_job_kwargs"} & calls[0][1].keys()
    assert calls[1][0] == step.prompt_format("query_execute", query=task.query) and "resume" not in calls[1][1]
    assert "send_chat_message_immediate" in calls[1][0]
    assert "send_chat_message_immediate" not in calls[1][1]["system_prompt"]
    assert ("## 任务准备摘要" if language == "zh" else "## Prepared task context") in calls[1][1]["system_prompt"]
    assert "Task context" in calls[1][1]["system_prompt"]
    assert frontmatter.load(env.runtime.workspace_path / saved.result_path).content == "调查结果。"
    assert saved.message_id == env.runtime.reminders.read()[0].id
    assert env.runtime.reminders.read()[0].content.startswith("【已延迟】") is delayed


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
    assert not env.runtime.temp_tasks.document_path(task.id, "result.md").exists()
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   ", "x" * 3981])
async def test_immediate_tool_does_not_checkpoint_invalid_results(pa_workspace, content):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("work")
    await env.runtime.temp_tasks.patch(task.id, task.revision, status="running")
    with pytest.raises(ValueError, match="1-3980"):
        await env.runtime.temp_tasks.send(task.id, task.revision, content)
    assert not env.runtime.temp_tasks.document_path(task.id, "result.md").exists()
    assert env.runtime.reminders.read() == []


@pytest.mark.asyncio
async def test_temporary_worker_recovers_checkpointed_result_without_model_call(pa_workspace, monkeypatch):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("work")
    task = await env.runtime.temp_tasks.save_text(task.id, task.revision, "handoff.md", "context", status="running")
    submit = env.runtime.reminders.submit_immediate
    monkeypatch.setattr(env.runtime.reminders, "submit_immediate", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await env.runtime.temp_tasks.send(task.id, task.revision, "saved result")
    saved = env.runtime.temp_tasks.get(task.id)
    assert saved.status == "sending" and saved.result_path
    monkeypatch.setattr(env.runtime.reminders, "submit_immediate", submit)
    env.agent.reply = AsyncMock(side_effect=AssertionError("Recovery must not reexecute the query"))

    await TempProactiveStep(app_context=env.context)._run(saved)

    assert env.runtime.temp_tasks.get(task.id).status == "completed"
    assert env.runtime.reminders.read()[0].content == "saved result"
    env.agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_temporary_worker_requires_result_tool_in_execution_turn(pa_workspace):
    env = pa_workspace
    task = await env.runtime.temp_tasks.create("work")
    task = await env.runtime.temp_tasks.save_text(task.id, task.revision, "handoff.md", "context", status="scheduled")
    env.agent.reply = AsyncMock(return_value={"status": "completed", "result": "Not submitted", "session_id": "worker"})
    env.agent.app_server.get_session_path = AsyncMock(return_value=None)

    await TempProactiveStep(app_context=env.context)._run(task)

    saved = env.runtime.temp_tasks.get(task.id)
    assert saved.status == "failed" and "send_chat_message_immediate" in saved.error
    env.agent.reply.assert_awaited_once()
    assert env.runtime.reminders.read() == []
