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
from reme.steps.proactive.tools import ReadProactiveInboxStep, WriteProactiveInboxStep
from reme.steps.proactive.api import ProactiveInboxViewStep, TaskResultsStep


@pytest.mark.asyncio
async def test_user_read_receipt_preserves_a_document_at_the_read_limit(pa_workspace, monkeypatch):
    env = pa_workspace
    path = env.runtime.tasks.task_path("one") / "2026-11-08" / "120000.md"
    post = frontmatter.Post("A useful result", task_id="one", status="unread")
    original = frontmatter.dumps(post) + "\n"
    env.runtime.files.write_text(path, original)
    monkeypatch.setattr("reme.components.pa.files.DEFAULT_MAX_FILE_BYTES", len(original.encode()))
    with pytest.raises(ValueError, match="read limit"):
        await TaskResultsStep(app_context=env.context)(
            action="read",
            path=env.runtime.to_workspace_relative(path),
            mark_read=True,
        )
    assert path.read_text() == original
    assert env.runtime.files.read(path) == original


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
        assert result.answer == f"{message['created_at']} {message['message']}"
        assert not (env.runtime.component_metadata_path / "proactive_context.jsonl").exists()
        again = await ReadProactiveInboxStep(app_context=env.context)(pa_round_id=current.id)
        assert again.answer == ""
    received = await view(action="list", kind="instruction")
    assert received.answer["pending"] == 0 and received.answer["items"][0]["status"] == "received"
    assert env.runtime.proactive_inbox_path.read_bytes() == source
    assert (await env.runtime.get_proactive_config()).interval_minutes == 5


@pytest.mark.asyncio
async def test_inbox_formats_messages_as_timestamp_and_content_without_changing_the_source(pa_workspace):
    env = pa_workspace
    first = await env.runtime.write_proactive_inbox(
        "关注黄金价格，只提醒重要变化。\n保留正文里的 Main Agent 和 ## 标题。",
    )
    env.now[0] += timedelta(minutes=5)
    second = await env.runtime.write_proactive_inbox("把唤醒间隔改成10分钟。")
    source = env.runtime.proactive_inbox_path.read_bytes()
    async with env.runtime.proactive_round() as current:
        response = await ReadProactiveInboxStep(app_context=env.context)(pa_round_id=current.id)
        assert response.answer == "\n\n".join(f"{item['created_at']} {item['message']}" for item in (first, second))
        assert first["id"] not in response.answer and second["id"] not in response.answer
        assert "· Main Agent" not in response.answer
        assert env.runtime.proactive_inbox_path.read_bytes() == source
        assert all(record["status"] == "consumed" for record in env.runtime.inbox.receipts().values())
        assert await env.runtime.read_proactive_inbox(round_id=current.id) == ""
        third = await env.runtime.write_proactive_inbox("新增关注原油。")
        assert (
            await env.runtime.read_proactive_inbox(round_id=current.id) == f"{third['created_at']} {third['message']}"
        )


@pytest.mark.asyncio
async def test_inbox_formats_an_edited_message_without_losing_its_new_body(pa_workspace):
    env = pa_workspace
    message = await env.runtime.write_proactive_inbox("原始反馈")
    path = env.runtime.proactive_inbox_path
    async with env.runtime.proactive_round() as current:
        await env.runtime.read_proactive_inbox(round_id=current.id)
        path.write_text(path.read_text(encoding="utf-8").replace("原始反馈", "修改后的反馈\n下一行"), encoding="utf-8")
        source = path.read_bytes()
        assert (
            await env.runtime.read_proactive_inbox(round_id=current.id)
            == f"{message['created_at']} 修改后的反馈\n下一行"
        )
    assert path.read_bytes() == source


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "完整反馈🙂\n" * 5000], ids=["empty", "long_unicode"])
async def test_registered_inbox_tool_returns_complete_text_directly(pa_workspace, content):
    env = pa_workspace
    env.runtime.proactive_inbox_path.write_text(content, encoding="utf-8")
    definition = resolve_app_config(config="proactive", log_config=False)["jobs"]["read_proactive_inbox"]
    job = BaseJob(name="read_proactive_inbox", app_context=env.context, **definition)
    async with job, env.runtime.proactive_round() as current:
        response = await job(pa_caller="proactive", pa_round_id=current.id)
    assert response.success and response.answer == content
    assert env.runtime.proactive_inbox_path.read_text(encoding="utf-8") == content


@pytest.mark.asyncio
async def test_inbox_returns_only_new_text_and_rereads_edited_source(pa_workspace):
    env = pa_workspace
    path = env.runtime.proactive_inbox_path
    original, appended, edited = "原始反馈🙂\n", "新增要求\n", "修改后的要求\n"
    path.write_text(original, encoding="utf-8")
    async with env.runtime.proactive_round() as current:
        assert await env.runtime.read_proactive_inbox(round_id=current.id) == original
        env.runtime.files.append_text(path, appended)
        assert await env.runtime.read_proactive_inbox(round_id=current.id) == appended
        path.write_text(edited, encoding="utf-8")
        assert await env.runtime.read_proactive_inbox(round_id=current.id) == edited
        assert await env.runtime.read_proactive_inbox(round_id=current.id) == ""
    assert path.read_text(encoding="utf-8") == edited
    assert not (env.runtime.component_metadata_path / "proactive_context.jsonl").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["read_limit", "cursor"])
async def test_inbox_read_failure_preserves_unread_content_and_receipts(pa_workspace, monkeypatch, failure):
    env = pa_workspace
    message = await env.runtime.write_proactive_inbox("完整保留这条反馈🙂")
    source = env.runtime.proactive_inbox_path.read_bytes()
    async with env.runtime.proactive_round() as current:
        with monkeypatch.context() as patch:
            kwargs = {"round_id": current.id}
            if failure == "read_limit":
                kwargs["max_bytes"] = len(source) - 1
            else:
                read = env.runtime.files.read

                def fail_cursor(path):
                    if path == env.runtime.inbox.cursor_path:
                        raise OSError("disk full")
                    return read(path)

                env.runtime.files.write_json(env.runtime.inbox.cursor_path, {"bytes": 0})
                patch.setattr(env.runtime.files, "read", fail_cursor)
            with pytest.raises((ValueError, OSError), match="read limit|disk full"):
                await env.runtime.read_proactive_inbox(**kwargs)
        if failure == "read_limit":
            assert not env.runtime.inbox.cursor_path.exists()
        else:
            assert env.runtime.inbox.consumed_bytes(source) == 0
        assert env.runtime.inbox.receipts()[message["id"]]["status"] == "queued"
        assert env.runtime.proactive_inbox_path.read_bytes() == source
        expected = f"{message['created_at']} {message['message']}"
        assert await env.runtime.read_proactive_inbox(round_id=current.id) == expected


@pytest.mark.asyncio
async def test_interrupted_publication_recovers_once_without_changing_settings(pa_workspace, monkeypatch):
    env = pa_workspace
    append = env.runtime.files.append_text
    before = env.runtime.proactive_config_path.read_bytes()

    def fail_source(path, content):
        if path == env.runtime.proactive_inbox_path:
            raise OSError("disk full")
        append(path, content)

    monkeypatch.setattr(env.runtime.files, "append_text", fail_source)
    with pytest.raises(OSError, match="disk full"):
        await env.runtime.write_proactive_inbox("把唤醒间隔改成10分钟")
    assert len(env.runtime.inbox.receipts()) == 1
    assert next(iter(env.runtime.inbox.receipts().values()))["status"] == "staging"
    monkeypatch.setattr(env.runtime.files, "append_text", append)
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
        env.runtime.files.write_json(env.runtime.proactive_config_path, {**settings, "enabled": False})
    elif mode == "outside_hours":
        env.now[0] = env.now[0].replace(hour=23)
    elif mode == "invalid":
        env.runtime.proactive_config_path.write_text("invalid JSON")
    message = "恢复主动提醒，关注黄金价格，有波动提醒我；把唤醒间隔改成10分钟"
    await env.runtime.write_proactive_inbox(message)
    before = env.runtime.proactive_config_path.read_bytes()
    if mode != "active":
        env.agent.reply = AsyncMock()
        step = ProactiveAgentStep(app_context=env.context, language=language)
        if mode == "invalid":
            with pytest.raises(ValueError):
                await step()
        else:
            result = await step()
            assert result.answer["reason"] == {"disabled": "disabled", "outside_hours": "outside_active_hours"}[mode]
        env.agent.create_session.assert_not_awaited()
        env.agent.compact_session.assert_not_awaited()
        env.agent.reply.assert_not_awaited()
        assert env.runtime.proactive_config_path.read_bytes() == before
        return
    queries = []

    async def reply(query, **_kwargs):
        queries.append(query)
        current = env.runtime.current_round
        result = await ReadProactiveInboxStep(app_context=env.context)(pa_round_id=current.id)
        assert message in result.answer
        assert env.runtime.proactive_config_path.read_bytes() == before
        await env.runtime.send_chat_message(
            "",
            kind="normal",
            round_id=current.id,
            session_id=current.session_id,
        )
        env.runtime.files.write_json(
            env.runtime.proactive_config_path,
            {**settings, "enabled": True, "interval_minutes": 10},
        )
        return {"session_id": "proactive", "status": "completed", "result": "No pending TODOs"}

    env.agent.reply = AsyncMock(side_effect=reply)
    tools = ["read_proactive_inbox", "check_json", "send_chat_message", "screen_capture", "read_cron_results"]
    step = ProactiveAgentStep(app_context=env.context, language=language, job_tools=tools)
    result = await step()
    assert result.success and len(queries) == 1
    config = await env.runtime.get_proactive_config()
    assert config.interval_minutes == 10
    assert config.next_wakeup(env.now[0]) == env.now[0] + timedelta(minutes=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disabled", "outside_hours", "invalid"])
async def test_loop_leaves_explicit_feedback_queued_while_notifications_are_stopped(pa_workspace, monkeypatch, mode):
    env = pa_workspace
    settings = (await env.runtime.get_proactive_config()).model_dump(mode="json")
    if mode == "disabled":
        env.runtime.files.write_json(env.runtime.proactive_config_path, {**settings, "enabled": False})
    elif mode == "outside_hours":
        env.now[0] = env.now[0].replace(hour=23)
    else:
        env.runtime.proactive_config_path.write_text("invalid JSON")
    await env.runtime.write_proactive_inbox("恢复提醒并修复配置")
    stop = asyncio.Event()

    run = AsyncMock()
    monkeypatch.setattr(PaLoopStep, "run_job", run)

    async def wait_for_stop(awaitable, timeout):
        assert timeout == 30
        stop.set()
        return await awaitable

    monkeypatch.setattr("reme.steps.proactive.background._lifecycle.asyncio.wait_for", wait_for_stop)
    await PaLoopStep(app_context=env.context)(stop_event=stop)
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_settings_request_is_retried_after_its_message_was_consumed(pa_workspace):
    env = pa_workspace
    settings = (await env.runtime.get_proactive_config()).model_dump(mode="json")
    await env.runtime.write_proactive_inbox("调整主动提醒要求")
    fail = [True]

    async def reply(_query, **_kwargs):
        current = env.runtime.current_round
        await env.runtime.read_proactive_inbox(round_id=current.id)
        if fail[0]:
            raise RuntimeError("model unavailable")
        env.runtime.files.write_json(env.runtime.proactive_config_path, {**settings, "enabled": True})
        await env.runtime.send_chat_message(
            "",
            kind="normal",
            round_id=current.id,
            session_id=current.session_id,
        )
        return {"session_id": "agent", "status": "completed", "result": "Processed feedback"}

    env.agent.reply = AsyncMock(side_effect=reply)
    with pytest.raises(RuntimeError, match="model unavailable"):
        await ProactiveAgentStep(app_context=env.context)()
    assert next(iter(env.runtime.inbox.receipts().values()))["status"] == "consumed"
    fail[0] = False
    await ProactiveAgentStep(app_context=env.context)()
