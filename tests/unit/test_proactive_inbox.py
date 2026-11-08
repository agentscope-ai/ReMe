"""Message-only feedback, durable consumption and agent-owned configuration changes."""

# pylint: disable=missing-function-docstring,protected-access

from datetime import timedelta
import asyncio
from unittest.mock import AsyncMock

import frontmatter
import pytest

from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.steps.proactive import ProactiveAgentStep, PaLoopStep
from reme.steps.proactive.tools import (
    ReadProactiveInboxStep,
    WriteProactiveInboxStep,
    ProactiveInboxViewStep,
    TaskResultsStep,
)


@pytest.mark.asyncio
async def test_user_read_receipt_preserves_a_document_at_the_read_limit(pa_workspace, monkeypatch):
    env = pa_workspace
    path = env.runtime.tasks.task_path("one") / "2026-11-08" / "120000.md"
    post = frontmatter.Post("A useful result", task_id="one", status="unread")
    original = frontmatter.dumps(post) + "\n"
    env.runtime._write_text(path, original)
    monkeypatch.setattr("reme.components._pa_tasks.DEFAULT_MAX_FILE_BYTES", len(original.encode()))
    with pytest.raises(ValueError, match="read limit"):
        await TaskResultsStep(app_context=env.context)(
            action="read",
            path=env.runtime.to_workspace_relative(path),
            mark_read=True,
        )
    assert path.read_text() == original
    assert env.runtime.tasks.read(path) == original


@pytest.mark.asyncio
async def test_registered_write_only_queues_message_without_configuration_access(pa_workspace):
    env = pa_workspace
    definition = resolve_app_config(config="proactive", log_config=False)["jobs"]["write_proactive_inbox"]
    assert set(definition["parameters"]["properties"]) == {"message"}
    message = "关注黄金价格，有波动提醒我；把唤醒间隔改成10分钟"
    before = env.runtime.proactive_config_path.read_bytes()
    allowed = [str(env.runtime.proactive_inbox_path), str(env.runtime.inbox.receipts_path)]
    job = BaseJob(name="write_proactive_inbox", app_context=env.context, **definition)
    async with job:
        response = await job(message=message, pa_caller="chat", _allowed_paths=allowed)
    assert response.success and response.answer["status"] == "queued"
    assert response.answer["message"] == message
    assert message in env.runtime.proactive_inbox_path.read_text()
    assert env.runtime.proactive_config_path.read_bytes() == before
    assert env.runtime.inbox.receipts()[response.answer["id"]]["status"] == "queued"
    assert "config" not in response.answer and "applied_at" not in response.answer
    assert env.runtime.current_round is None


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [None, "", " \n ", 10, True])
async def test_invalid_message_leaves_inbox_and_receipts_unchanged(pa_workspace, message):
    env = pa_workspace
    before = (env.runtime.proactive_inbox_path.read_bytes(), env.runtime.inbox.receipts_path.read_bytes())
    with pytest.raises(ValueError, match="non-empty string"):
        await env.runtime.write_proactive_inbox(message)
    assert (env.runtime.proactive_inbox_path.read_bytes(), env.runtime.inbox.receipts_path.read_bytes()) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [[], ["../outside"], ["source"], ["receipts"]])
async def test_write_requires_access_to_source_and_receipts(pa_workspace, scope):
    env = pa_workspace
    paths = {"source": str(env.runtime.proactive_inbox_path), "receipts": str(env.runtime.inbox.receipts_path)}
    allowed = [paths.get(name, name) for name in scope]
    with pytest.raises(PermissionError):
        await WriteProactiveInboxStep(app_context=env.context)(
            pa_caller="chat",
            message="关注黄金",
            _allowed_paths=allowed,
        )
    assert env.runtime.proactive_inbox_path.read_text() == "" and env.runtime.inbox.receipts() == {}


@pytest.mark.asyncio
async def test_only_main_can_write_feedback(pa_workspace):
    env = pa_workspace
    with pytest.raises(ValueError, match="bound to Main"):
        await WriteProactiveInboxStep(app_context=env.context)(pa_caller="proactive", message="修改间隔")
    assert env.runtime.inbox.receipts() == {}


@pytest.mark.asyncio
async def test_consumption_preserves_source_and_desktop_reading_does_not_consume(pa_workspace):
    env = pa_workspace
    message = await env.runtime.write_proactive_inbox("把唤醒间隔改成10分钟")
    source = env.runtime.proactive_inbox_path.read_bytes()
    view = ProactiveInboxViewStep(app_context=env.context)
    pending = await view(action="list", kind="instruction")
    assert pending.answer["pending"] == 1 and pending.answer["items"][0]["status"] == "pending"
    assert env.runtime.inbox.receipts()[message["id"]]["status"] == "queued"
    assert not env.runtime.inbox.cursor_path.exists()
    async with env.runtime.proactive_round() as current:
        result = await ReadProactiveInboxStep(app_context=env.context)(pa_round_id=current.id)
        checkpoint = env.runtime.workspace_path / result.answer["input_path"]
        assert message["message"] in checkpoint.read_text()
        again = await ReadProactiveInboxStep(app_context=env.context)(pa_round_id=current.id)
        assert again.answer["input_path"] is None
    received = await view(action="list", kind="instruction")
    assert received.answer["pending"] == 0 and received.answer["items"][0]["status"] == "received"
    assert env.runtime.proactive_inbox_path.read_bytes() == source
    assert (await env.runtime.get_proactive_config()).interval_minutes == 5


@pytest.mark.asyncio
async def test_interrupted_publication_recovers_once_without_changing_settings(pa_workspace, monkeypatch):
    env = pa_workspace
    append = env.runtime._append_text
    before = env.runtime.proactive_config_path.read_bytes()

    def fail_source(path, content):
        if path == env.runtime.proactive_inbox_path:
            raise OSError("disk full")
        append(path, content)

    monkeypatch.setattr(env.runtime, "_append_text", fail_source)
    with pytest.raises(OSError, match="disk full"):
        await env.runtime.write_proactive_inbox("把唤醒间隔改成10分钟")
    assert len(env.runtime.inbox.receipts()) == 1
    assert next(iter(env.runtime.inbox.receipts().values()))["status"] == "staging"
    monkeypatch.setattr(env.runtime, "_append_text", append)
    env.runtime.inbox.initialize()
    recovered = env.runtime.proactive_inbox_path.read_bytes()
    env.runtime.inbox.initialize()
    assert env.runtime.proactive_inbox_path.read_bytes() == recovered
    assert recovered.decode().count("把唤醒间隔改成10分钟") == 1
    assert next(iter(env.runtime.inbox.receipts().values()))["status"] == "queued"
    assert env.runtime.proactive_config_path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("mode", ["active", "disabled", "outside_hours", "invalid"])
async def test_proactive_agent_handles_message_then_changes_its_own_settings(pa_workspace, language, mode):
    env = pa_workspace
    settings = (await env.runtime.get_proactive_config()).model_dump(mode="json")
    if mode == "disabled":
        env.runtime._write_json(env.runtime.proactive_config_path, {**settings, "enabled": False})
    elif mode == "outside_hours":
        env.now[0] = env.now[0].replace(hour=23)
    elif mode == "invalid":
        env.runtime.proactive_config_path.write_text("invalid JSON")
    message = "恢复主动提醒，关注黄金价格，有波动提醒我；把唤醒间隔改成10分钟"
    await env.runtime.write_proactive_inbox(message)
    before = env.runtime.proactive_config_path.read_bytes()
    phases = []
    todos = []

    async def reply(query, **kwargs):
        if kwargs.get("fork_session"):
            return {"session_id": "context", "status": "completed", "result": "Main context"}
        current = env.runtime.current_round
        phases.append(current.phase)
        if current.phase == "query_memory":
            result = await ReadProactiveInboxStep(app_context=env.context)(pa_round_id=current.id)
            checkpoint = env.runtime.workspace_path / result.answer["input_path"]
            assert message in checkpoint.read_text()
            assert env.runtime.proactive_config_path.read_bytes() == before
        elif current.phase == "query_message":
            await env.runtime.send_chat_message(
                "",
                kind="normal",
                round_id=current.id,
                session_id=current.session_id,
            )
        elif current.phase == "query_config":
            assert env.runtime.proactive_config_path.read_bytes() == before
            values = {**settings, "enabled": True, "interval_minutes": 10}
            env.runtime._write_json(env.runtime.proactive_config_path, values)
        elif current.phase == "query_todo":
            todos.append(query)
        return {"session_id": "proactive", "status": "completed", "result": "No pending TODOs"}

    env.agent.reply = AsyncMock(side_effect=reply)
    step = ProactiveAgentStep(app_context=env.context, language=language)
    result = await step()
    assert result.success and phases.index("query_memory") < phases.index("query_config")
    config = await env.runtime.get_proactive_config()
    assert config.interval_minutes == 10
    if mode == "active":
        assert config.next_wakeup(env.now[0]) == env.now[0] + timedelta(minutes=10)
    else:
        assert phases == ["query_memory", "query_config", "query_todo"]
    assert todos == [step.prompt_format("query_todo", interval_minutes=10, todo_path=env.runtime.proactive_todo_path)]
    assert env.runtime.reminders.read() == [] and not env.runtime.inbox.pending()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disabled", "outside_hours", "invalid"])
async def test_loop_processes_explicit_feedback_while_notifications_are_stopped(pa_workspace, monkeypatch, mode):
    env = pa_workspace
    settings = (await env.runtime.get_proactive_config()).model_dump(mode="json")
    if mode == "disabled":
        env.runtime._write_json(env.runtime.proactive_config_path, {**settings, "enabled": False})
    elif mode == "outside_hours":
        env.now[0] = env.now[0].replace(hour=23)
    else:
        env.runtime.proactive_config_path.write_text("invalid JSON")
    await env.runtime.write_proactive_inbox("恢复提醒并修复配置")
    stop = asyncio.Event()

    async def run(_step, name, **kwargs):
        assert name == "pa_proactive" and kwargs["stop_event"] is stop
        stop.set()

    monkeypatch.setattr(PaLoopStep, "run_job", run)
    await PaLoopStep(app_context=env.context)(stop_event=stop)
    assert stop.is_set()


@pytest.mark.asyncio
async def test_failed_settings_request_is_retried_after_its_message_was_consumed(pa_workspace):
    env = pa_workspace
    settings = (await env.runtime.get_proactive_config()).model_dump(mode="json")
    env.runtime._write_json(env.runtime.proactive_config_path, {**settings, "enabled": False})
    await env.runtime.write_proactive_inbox("恢复主动提醒")
    fail = [True]

    async def reply(_query, **kwargs):
        if not kwargs.get("fork_session"):
            current = env.runtime.current_round
            if current.phase == "query_memory":
                await env.runtime.read_proactive_inbox(round_id=current.id)
            elif current.phase == "query_config":
                if fail[0]:
                    raise RuntimeError("model unavailable")
                env.runtime._write_json(env.runtime.proactive_config_path, {**settings, "enabled": True})
        return {"session_id": "agent", "status": "completed", "result": "Processed feedback"}

    env.agent.reply = AsyncMock(side_effect=reply)
    with pytest.raises(RuntimeError, match="model unavailable"):
        await ProactiveAgentStep(app_context=env.context)()
    assert env.runtime.inbox.pending()
    assert next(iter(env.runtime.inbox.receipts().values()))["status"] == "consumed"
    fail[0] = False
    await ProactiveAgentStep(app_context=env.context)()
    assert (await env.runtime.get_proactive_config()).enabled and not env.runtime.inbox.pending()
