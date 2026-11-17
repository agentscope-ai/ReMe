"""Proactive rounds with real workspace files and jobs at mocked model/session boundaries."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import frontmatter
import pytest
import pytest_asyncio

from reme.components.agent_wrapper import CodexAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import LocalFileCatalog
from reme.components.job import BackgroundJob, BaseJob
from reme.config import resolve_app_config
from reme.schema import Response, SessionMessage, SessionSnapshot, SessionStatus
from reme.steps.proactive import FindInterestStep, PaLoopStep, PushInterestStep
from reme.steps.proactive._interest import _InterestStep
from reme.steps.proactive.loop import WAKEUP_KEY
from reme.steps.proactive.pa import PaNewSessionStep
from reme.steps.proactive.tools import SendReminderStep, SetNextWakeupStep


@pytest_asyncio.fixture
async def runtime(tmp_path, monkeypatch):
    context = ApplicationContext(
        workspace_dir=str(tmp_path),
        daily_dir="notes",
        digest_dir="summaries",
        session_dir="chats",
        language="zh",
    )
    agent = CodexAgentWrapper(app_context=context, app_server="")
    agent.reply = AsyncMock(return_value={"status": "completed"})
    native_path = tmp_path / ".codex/sessions/current.jsonl"
    native_path.parent.mkdir(parents=True)
    native_path.write_text('{"role":"user","content":"记忆研究有什么进展？"}\n')
    snapshot = SessionSnapshot(
        session_id="session-1",
        state=SessionStatus(session_id="session-1", status="idle"),
        path=str(native_path),
        last_turn_id="turn-1",
    )
    agent.get_session_path = AsyncMock(return_value=native_path)
    agent.read_session = AsyncMock(return_value=snapshot)
    agent.get_session_status = AsyncMock(side_effect=lambda _session_id: snapshot.state)

    async def append(_session_id, content, *, message_id, **_kwargs):
        message = SessionMessage(
            id=f"reme-proactive-{message_id}",
            role="assistant",
            content=content,
            source="proactive",
            created_at="2026-11-05T14:30:13+08:00",
        )
        snapshot.messages.append(message)
        return message

    agent.append_session = AsyncMock(side_effect=append)
    context.components["agent_wrapper"] = {"default": agent}
    context.metadata["pa"] = {"session_id": "session-1", "lock": asyncio.Lock()}
    memory = tmp_path / "summaries/personal/REME_MEMORY.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("主人关注 ReMe 和记忆研究。", encoding="utf-8")
    instant = datetime(2026, 11, 5, 14, 30, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(_InterestStep, "_now", lambda self: instant)
    jobs = resolve_app_config(config="proactive")["jobs"]
    for name in ("send_reminder", "set_next_wakeup", "session_status"):
        job = BaseJob(name=name, app_context=context, **jobs[name])
        context.jobs[name] = job
        await job.start()
    yield context, agent
    for job in context.jobs.values():
        await job.close()


def note(context, name="interest_090000.md", **metadata):
    path = Path(context.app_config.workspace_dir) / context.app_config.daily_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.dumps(frontmatter.Post("值得关注的研究发现。", **metadata)), encoding="utf-8")
    return path


def change(path, content=None, **metadata):
    post = frontmatter.loads(path.read_text(encoding="utf-8"))
    post.metadata.update(metadata)
    if content is not None:
        post.content = content
    path.write_text(frontmatter.dumps(post), encoding="utf-8")


async def push(context):
    return await PushInterestStep(app_context=context, default_wakeup_minutes=7)()


async def send(context, kwargs, files, message="主人，有一项相关研究值得看看。"):
    return await context.jobs["send_reminder"](
        message=message,
        files=files,
        **kwargs["injected_job_kwargs"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("step_cls", [FindInterestStep, PushInterestStep])
@pytest.mark.parametrize("content", [None, "", "---\ntitle: empty\n---\n"])
async def test_missing_memory_is_initialized_and_existing_empty_memory_does_not_skip_the_agent(
    runtime,
    step_cls,
    content,
):
    context, agent = runtime
    memory = Path(context.app_config.workspace_dir) / "summaries/personal/REME_MEMORY.md"
    if content is None:
        memory.unlink()
    else:
        memory.write_text(content)
    note(context)
    step = step_cls(app_context=context, default_wakeup_minutes=7)
    result = await step()
    assert result.success
    assert memory.read_text(encoding="utf-8") == (
        step.get_prompt("default_memory") + "\n" if content is None else content
    )
    agent.reply.assert_awaited_once()
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("step_cls", [FindInterestStep, PushInterestStep])
async def test_memory_initialization_errors_propagate_without_calling_the_agent(runtime, monkeypatch, step_cls):
    context, agent = runtime
    memory = Path(context.app_config.workspace_dir) / "summaries/personal/REME_MEMORY.md"
    memory.unlink()
    monkeypatch.setattr(
        "reme.steps.proactive.pa.write_file_safe",
        AsyncMock(side_effect=PermissionError("write denied")),
    )
    with pytest.raises(PermissionError, match="write denied"):
        await step_cls(app_context=context)()
    assert not memory.exists()
    agent.reply.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("step_cls", [FindInterestStep, PushInterestStep])
async def test_proactive_parameters_are_validated_before_initializing_memory(runtime, step_cls):
    context, agent = runtime
    memory = Path(context.app_config.workspace_dir) / "summaries/personal/REME_MEMORY.md"
    memory.unlink()
    with pytest.raises(ValueError, match="Unexpected PA parameters"):
        await step_cls(app_context=context)(query="unexpected")
    assert not memory.exists()
    agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_foreground_and_background_share_one_memory_initialization(runtime, monkeypatch):
    from reme.steps.proactive import pa

    context, agent = runtime
    memory = Path(context.app_config.workspace_dir) / "summaries/personal/REME_MEMORY.md"
    memory.unlink()
    entered, release = asyncio.Event(), asyncio.Event()
    write = pa.write_file_safe

    async def hold_write(*args, **kwargs):
        entered.set()
        await release.wait()
        await write(*args, **kwargs)

    writer = AsyncMock(side_effect=hold_write)
    monkeypatch.setattr(pa, "write_file_safe", writer)
    agent.create_session = AsyncMock(return_value="session-2")
    agent.release_session = AsyncMock()
    collector = FindInterestStep(app_context=context)
    background = asyncio.create_task(collector())
    await entered.wait()
    foreground = asyncio.create_task(
        PaNewSessionStep(app_context=context, prompt_dict={"default_memory_zh": "不应覆盖首次初始化。"})(),
    )
    try:
        await asyncio.sleep(0)
        assert not foreground.done()
    finally:
        release.set()
        await asyncio.gather(background, foreground)
    writer.assert_awaited_once()
    assert memory.read_text(encoding="utf-8") == collector.get_prompt("default_memory") + "\n"
    assert context.metadata["pa"]["session_id"] == "session-2"


@pytest.mark.asyncio
async def test_collector_saves_before_ack_and_code_initializes_only_output_metadata(runtime):
    context, agent = runtime
    workspace = Path(context.app_config.workspace_dir)
    catalog = LocalFileCatalog(app_context=context, watch_dirs=["resource_dir"], watch_filenames=["metadata.json"])
    context.components["file_catalog"] = {"resource": catalog}
    resource = workspace / "resource/2026-11-05/package/metadata.json"
    resource.parent.mkdir(parents=True)
    resource.write_text('{"description":"memory research","files":[]}')
    config = resolve_app_config(config="proactive")
    for name in ("resource_changes", "resource_processed"):
        job = BaseJob(name=name, app_context=context, **config["jobs"][name])
        context.jobs[name] = job
        await job.start()
    memory = workspace / "summaries/personal/REME_MEMORY.md"
    original = memory.read_bytes()

    async def discover(_inputs, **kwargs):
        assert kwargs["job_tools"] == ["resource_changes", "resource_processed"]
        assert "resume" not in kwargs
        prompt = kwargs["system_prompt"]
        assert str(memory) in prompt and str(workspace / "summaries/personal/INTERESTS.md") in prompt
        assert "目标 PA 会话：session-1" in prompt
        assert f"只读会话文件：{agent.get_session_path.return_value}" in prompt
        paths = (await context.jobs["resource_changes"]()).answer
        target = workspace / "notes/interest_143012.md"
        assert str(target) in prompt
        target.parent.mkdir(parents=True)
        target.write_text(frontmatter.dumps(frontmatter.Post("研究发现。", sources=paths)))
        (memory.parent / "INTERESTS.md").write_text("# 兴趣观察\n\n## 桌面机器人\n- 待确认，来自主人当前项目。")
        assert (await context.jobs["resource_processed"](paths=paths)).success
        assert "status" not in frontmatter.loads(target.read_text())
        return {"status": "completed"}

    agent.reply.side_effect = discover
    result = await FindInterestStep(app_context=context)()
    assert result.answer == {"created": ["notes/interest_143012.md"]}
    post = frontmatter.loads((workspace / result.answer["created"][0]).read_text())
    assert post["status"] == "unsent" and post["sources"] == [resource.relative_to(workspace).as_posix()]
    assert post["created_at"] == post["updated_at"] == "2026-11-05T14:30:12+08:00"
    assert await catalog.changes() == []
    assert memory.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("session_path", ["available", "unavailable"])
async def test_collector_receives_current_session_context_without_guessing_a_missing_path(
    runtime,
    language,
    session_path,
):
    context, agent = runtime
    context.app_config.language = language
    native_path = agent.get_session_path.return_value
    if session_path == "unavailable":
        agent.get_session_path.return_value = None
    await FindInterestStep(app_context=context)()
    prompt = agent.reply.call_args.kwargs["system_prompt"]
    agent.get_session_path.assert_awaited_once_with("session-1")
    agent.read_session.assert_not_awaited()
    assert "session-1" in prompt
    if session_path == "available":
        assert str(native_path) in prompt
    else:
        assert str(native_path) not in prompt
        assert ("只读会话文件：\n" if language == "zh" else "Read-only session file: \n") in prompt


@pytest.mark.asyncio
async def test_collector_skips_without_a_current_session(runtime):
    context, agent = runtime
    context.metadata["pa"]["session_id"] = None
    memory = Path(context.app_config.workspace_dir) / "summaries/personal/REME_MEMORY.md"
    memory.unlink()
    assert (await FindInterestStep(app_context=context)()).answer == {"created": []}
    assert "财经新闻" in memory.read_text(encoding="utf-8")
    agent.get_session_path.assert_not_awaited()
    agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_collector_does_not_start_with_a_superseded_session(runtime):
    context, agent = runtime

    async def switch_session(_session_id):
        context.metadata["pa"]["session_id"] = "session-2"
        return Path(agent.read_session.return_value.path)

    agent.get_session_path.side_effect = switch_session
    assert (await FindInterestStep(app_context=context)()).answer == {"created": []}
    agent.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_findings_and_filename_collisions_do_not_overwrite_existing_notes(runtime):
    context, agent = runtime
    existing = note(context, "interest_143012.md", status="sent")
    before = existing.read_bytes()
    result = await FindInterestStep(app_context=context)()
    assert result.answer == {"created": []}
    assert str(existing.with_name("interest_143012_1.md")) in agent.reply.call_args.kwargs["system_prompt"]
    assert existing.read_bytes() == before
    assert not existing.with_name("interest_143012_1.md").exists()


@pytest.mark.asyncio
async def test_reminder_agent_uses_bound_jobs_to_send_multiple_files_and_set_wakeup(runtime):
    context, agent = runtime
    paths = [note(context), note(context, "interest_100000.md")]
    receipts = []

    async def choose(_inputs, **kwargs):
        prompt = kwargs["system_prompt"]
        assert agent.get_session_path.return_value.as_posix() in prompt
        assert "7 分钟" in prompt
        assert kwargs["job_tools"] == ["session_status", "send_reminder", "set_next_wakeup"]
        assert "resume" not in kwargs
        state = await context.jobs["session_status"](**kwargs["injected_job_kwargs"])
        assert state.answer["session_id"] == "session-1"
        receipts.append(await send(context, kwargs, [p.name for p in paths]))
        receipts.append(await send(context, kwargs, [p.name for p in reversed(paths)]))
        await context.jobs["set_next_wakeup"](minutes=12, **kwargs["injected_job_kwargs"])
        return {"status": "completed"}

    agent.reply.side_effect = choose
    result = await push(context)
    assert result.answer["sent"] and receipts[0].answer == receipts[1].answer
    posts = [frontmatter.loads(path.read_text()) for path in paths]
    assert all(post["status"] == "sent" and post["target_session_id"] == "session-1" for post in posts)
    assert posts[0]["message_id"] == posts[1]["message_id"]
    assert context.metadata[WAKEUP_KEY] == 12
    assert not context.metadata["pa"]["rounds"]
    agent.get_session_path.assert_awaited_once_with("session-1")
    agent.append_session.assert_awaited_once_with(
        "session-1",
        "主人，有一项相关研究值得看看。",
        message_id=posts[0]["message_id"],
        timeout=1,
        expected_turn_id="turn-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["idle", "active", "error", "not_loaded"])
async def test_delivery_checks_activity_and_lets_owner_restore_unloaded_session(runtime, status):
    context, agent = runtime
    path = note(context)
    agent.read_session.return_value.state.status = status

    async def choose(_inputs, **kwargs):
        await send(context, kwargs, [path.name])
        return {"status": "completed"}

    agent.reply.side_effect = choose
    result = await push(context)
    expected = status in {"idle", "not_loaded"}
    assert result.answer["sent"] == expected
    assert frontmatter.loads(path.read_text())["status"] == ("sent" if expected else "unsent")
    assert agent.append_session.await_count == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["session", "turn", "lock"])
async def test_changes_during_the_decision_defer_delivery(runtime, mutation):
    context, agent = runtime
    path = note(context)

    async def choose(_inputs, **kwargs):
        if mutation == "session":
            context.metadata["pa"]["session_id"] = "session-2"
        elif mutation == "turn":
            agent.read_session.return_value.last_turn_id = "turn-2"
        else:
            await context.metadata["pa"]["lock"].acquire()
        try:
            result = await send(context, kwargs, [path.name])
            assert result.success and not result.answer["sent"]
        finally:
            if mutation == "lock":
                context.metadata["pa"]["lock"].release()
        return {"status": "completed"}

    agent.reply.side_effect = choose
    await push(context)
    assert frontmatter.loads(path.read_text())["status"] == "unsent"
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_owner_turn_guard_defers_and_allows_fresh_planning_next_round(runtime):
    from reme.components.codex_app_server import SessionChangedError

    context, agent = runtime
    path = note(context)
    original_append = agent.append_session.side_effect

    async def choose(_inputs, **kwargs):
        await send(context, kwargs, [path.name])
        return {"status": "completed"}

    agent.reply.side_effect = choose
    agent.append_session.side_effect = SessionChangedError("new turn")
    assert not (await push(context)).answer["sent"]
    first_id = agent.append_session.call_args.kwargs["message_id"]
    agent.append_session.side_effect = original_append
    assert (await push(context)).answer["sent"]
    assert agent.append_session.call_args.kwargs["message_id"] != first_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "files",
    [
        [],
        ["../interest_090000.md"],
        ["/interest_090000.md"],
        ["interest_256000.md"],
        ["interest_090000.md", "interest_090000.md"],
        ["interest_000000.md"],
    ],
)
async def test_invalid_file_batches_do_not_inject_or_change_valid_notes(runtime, files):
    context, agent = runtime
    path = note(context, status="unsent", created_at="original")
    original = path.read_bytes()

    async def choose(_inputs, **kwargs):
        result = await send(context, kwargs, files)
        assert not result.success
        return {"status": "completed"}

    agent.reply.side_effect = choose
    await push(context)
    assert path.read_bytes() == original
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["body", "metadata", "remove", "sent", "extra"])
async def test_selected_files_must_match_the_prepared_context(runtime, mutation):
    context, agent = runtime
    path = note(context)

    async def choose(_inputs, **kwargs):
        files = [path.name]
        if mutation == "body":
            change(path, content="主人修改后的正文")
        elif mutation == "metadata":
            change(path, sources=["changed"])
        elif mutation == "remove":
            path.unlink()
        elif mutation == "sent":
            change(path, status="sent")
        else:
            files = [note(context, "interest_100000.md", status="unsent").name]
        result = await send(context, kwargs, files)
        assert not result.success
        return {"status": "completed"}

    agent.reply.side_effect = choose
    await push(context)
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_unbound_reminder_job_cannot_select_a_session(runtime):
    context, agent = runtime
    path = note(context)
    result = await context.jobs["send_reminder"](message="提醒", files=[path.name], session_id="session-1")
    assert not result.success
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_tools_hide_binding_parameters_and_expire_with_the_round(runtime):
    from fastmcp import Client, FastMCP
    from fastmcp.exceptions import ToolError
    from reme.components.service.mcp_tools import add_mcp_job

    context, agent = runtime
    path = note(context)
    binding = {}

    async def choose(_inputs, **kwargs):
        binding.update(kwargs["injected_job_kwargs"])
        server = FastMCP("Remy test")
        for name in kwargs["job_tools"]:
            add_mcp_job(
                server,
                context.jobs[name],
                injected_job_kwargs=binding,
                tool_error_on_failure=True,
            )
        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            assert set(tools["send_reminder"].input_schema["properties"]) == {"message", "files"}
            assert tools["session_status"].input_schema["properties"] == {}
            assert (await client.call_tool("session_status", {})).data["session_id"] == "session-1"
            with pytest.raises(ToolError):
                await client.call_tool(
                    "send_reminder",
                    {"message": "提醒", "files": [path.name], "session_id": "other"},
                )
            result = await client.call_tool("send_reminder", {"message": "提醒", "files": [path.name]})
            assert result.data["sent"]
        return {"status": "completed"}

    agent.reply.side_effect = choose
    await push(context)
    expired = await context.jobs["send_reminder"](message="提醒", files=[path.name], **binding)
    assert not expired.success and agent.append_session.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [None, "", " ", "x" * 100001])
async def test_invalid_reminder_text_leaves_files_unsent(runtime, message):
    context, agent = runtime
    path = note(context)

    async def choose(_inputs, **kwargs):
        assert not (await send(context, kwargs, [path.name], message)).success
        return {"status": "completed"}

    agent.reply.side_effect = choose
    await push(context)
    assert frontmatter.loads(path.read_text())["status"] == "unsent"
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_can_defer_and_cannot_request_two_different_reminders(runtime):
    context, agent = runtime
    paths = [note(context), note(context, "interest_100000.md")]
    await push(context)
    agent.append_session.assert_not_awaited()

    async def choose(_inputs, **kwargs):
        assert (await send(context, kwargs, [paths[0].name])).success
        assert not (await send(context, kwargs, [paths[1].name])).success
        return {"status": "completed"}

    agent.reply.side_effect = choose
    await push(context)
    assert agent.append_session.await_count == 1
    assert frontmatter.loads(paths[1].read_text())["status"] == "unsent"


@pytest.mark.asyncio
async def test_status_write_failure_does_not_block_other_notes_or_later_rounds(runtime, monkeypatch):
    context, agent = runtime
    paths = [note(context), note(context, "interest_100000.md"), note(context, "interest_110000.md")]
    save = SendReminderStep._save_note
    failed = False

    async def interrupted(self, filename, post):
        nonlocal failed
        if filename == paths[1].name and not failed:
            failed = True
            raise OSError("interrupted status update")
        await save(self, filename, post)

    async def choose(_inputs, **kwargs):
        files = [p.name for p in paths if frontmatter.loads(p.read_text())["status"] == "unsent"]
        result = await send(context, kwargs, files)
        assert result.success and result.answer["sent"]
        return {"status": "completed"}

    monkeypatch.setattr(SendReminderStep, "_save_note", interrupted)
    agent.reply.side_effect = choose
    await push(context)
    assert frontmatter.loads(paths[0].read_text())["status"] == "sent"
    assert frontmatter.loads(paths[1].read_text())["status"] == "unsent"
    assert frontmatter.loads(paths[2].read_text())["status"] == "sent"
    context.metadata["pa"]["session_id"] = "session-2"
    result = await push(context)
    assert result.answer["sent"]
    assert all(frontmatter.loads(p.read_text())["status"] == "sent" for p in paths)
    assert agent.reply.await_count == agent.append_session.await_count == 2
    assert not (Path(context.app_config.workspace_dir) / "chats/proactive/pending.json").exists()


@pytest.mark.asyncio
async def test_retry_within_one_round_reuses_message_id_and_caches_success(runtime):
    context, agent = runtime
    path = note(context)
    original_append = agent.append_session.side_effect

    async def choose(_inputs, **kwargs):
        agent.append_session.side_effect = TimeoutError("publication interrupted")
        assert not (await send(context, kwargs, [path.name])).success
        message_id = agent.append_session.call_args.kwargs["message_id"]
        assert frontmatter.loads(path.read_text())["status"] == "unsent"
        agent.append_session.side_effect = original_append
        delivered = await send(context, kwargs, [path.name])
        repeated = await send(context, kwargs, [path.name])
        assert delivered.success and repeated.answer == delivered.answer
        assert delivered.answer["message_id"] == message_id
        return {"status": "completed"}

    agent.reply.side_effect = choose
    assert (await push(context)).answer["sent"]
    assert agent.reply.await_count == 1
    assert agent.append_session.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_interrupted_append_allows_fresh_planning_next_round(runtime, accepted):
    context, agent = runtime
    path = note(context)
    original_append = agent.append_session.side_effect

    async def interrupted(*args, **kwargs):
        if accepted:
            await original_append(*args, **kwargs)
        raise TimeoutError("publication interrupted")

    async def choose(_inputs, **kwargs):
        await send(context, kwargs, [path.name])
        return {"status": "completed"}

    agent.reply.side_effect = choose
    agent.append_session.side_effect = interrupted
    assert not (await push(context)).answer["sent"]
    original_id = agent.append_session.call_args.kwargs["message_id"]
    assert frontmatter.loads(path.read_text())["status"] == "unsent"
    change(path, content="主人编辑后的内容")
    agent.append_session.side_effect = original_append
    assert (await push(context)).answer["sent"]
    assert agent.reply.await_count == agent.append_session.await_count == 2
    assert agent.append_session.call_args.kwargs["message_id"] != original_id
    assert "主人编辑后的内容" in path.read_text()
    assert frontmatter.loads(path.read_text())["status"] == "sent"
    assert len(agent.read_session.return_value.messages) == 1 + accepted


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["body", "metadata", "remove"])
async def test_concurrent_edits_preserve_notes_and_allow_later_reminders(runtime, mutation):
    context, agent = runtime
    path = note(context)
    companion = note(context, "interest_100000.md")
    original_append = agent.append_session.side_effect
    edited = None

    async def edit_during_append(*args, **kwargs):
        nonlocal edited
        result = await original_append(*args, **kwargs)
        if mutation == "remove":
            path.unlink()
        else:
            if mutation == "body":
                change(path, content="注入期间主人编辑后的内容")
            else:
                change(path, sources=["主人补充的来源"])
            edited = path.read_bytes()
        return result

    async def choose(_inputs, **kwargs):
        result = await send(context, kwargs, [path.name, companion.name])
        assert result.success and result.answer["sent"]
        assert (await send(context, kwargs, [path.name, companion.name])).answer == result.answer
        return {"status": "completed"}

    agent.reply.side_effect = choose
    agent.append_session.side_effect = edit_during_append
    assert (await push(context)).answer["sent"]
    assert frontmatter.loads(companion.read_text())["status"] == "sent"
    assert agent.append_session.await_count == 1
    if edited is None:
        assert not path.exists()
    else:
        assert path.read_bytes() == edited
        assert frontmatter.loads(path.read_text())["status"] == "unsent"
    later = note(context, "interest_110000.md", status="unsent")
    context.metadata["pa"]["session_id"] = "session-2"
    agent.append_session.side_effect = original_append

    async def choose_later(_inputs, **kwargs):
        await send(context, kwargs, [later.name])
        return {"status": "completed"}

    agent.reply.side_effect = choose_later
    assert (await push(context)).answer["sent"]
    assert agent.reply.await_count == agent.append_session.await_count == 2
    assert frontmatter.loads(later.read_text())["status"] == "sent"
    assert agent.append_session.call_args.args[0] == "session-2"
    if edited is not None:
        assert path.read_bytes() == edited


@pytest.mark.asyncio
async def test_cancellation_finishes_accepted_delivery_and_status_updates(runtime):
    context, agent = runtime
    path = note(context)
    entered, release = asyncio.Event(), asyncio.Event()
    original_append = agent.append_session.side_effect

    async def hold_append(*args, **kwargs):
        result = await original_append(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    async def choose(_inputs, **kwargs):
        await send(context, kwargs, [path.name])
        return {"status": "completed"}

    agent.append_session.side_effect = hold_append
    agent.reply.side_effect = choose
    task = asyncio.create_task(push(context))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert frontmatter.loads(path.read_text())["status"] == "sent"
    assert not context.metadata["pa"]["lock"].locked()
    assert not context.metadata["pa"]["rounds"]


@pytest.mark.asyncio
async def test_round_revisits_older_candidates_without_new_findings(runtime):
    context, agent = runtime
    path = note(context)
    calls = []

    async def respond(_inputs, **kwargs):
        calls.append(kwargs["job_tools"])
        if "send_reminder" in kwargs["job_tools"]:
            await send(context, kwargs, [path.name])
        return {"status": "completed"}

    agent.reply.side_effect = respond
    job = BaseJob(
        name="pa_proactive",
        app_context=context,
        **resolve_app_config(config="proactive")["jobs"]["pa_proactive"],
    )
    async with job:
        result = await job(stop_event=asyncio.Event())
    assert result.success and result.answer["sent"]
    assert calls == [
        ["resource_changes", "resource_processed"],
        ["session_status", "send_reminder", "set_next_wakeup"],
    ]


@pytest.mark.asyncio
async def test_no_session_or_candidates_skips_the_reminder_agent(runtime):
    context, agent = runtime
    await push(context)
    note(context)
    context.metadata["pa"]["session_id"] = None
    await push(context)
    agent.reply.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes", [None, 0, float("inf")])
async def test_invalid_configured_delay_fails_before_running_the_reminder_agent(runtime, minutes):
    context, agent = runtime
    note(context)
    with pytest.raises(ValueError, match="positive finite"):
        await PushInterestStep(app_context=context, default_wakeup_minutes=minutes)()
    agent.reply.assert_not_awaited()
    assert not context.metadata["pa"]["rounds"]


@pytest.mark.asyncio
async def test_interest_scan_respects_the_configured_file_byte_limit(runtime):
    context, agent = runtime
    note(context)
    with pytest.raises(ValueError, match="exceeds read limit"):
        await PushInterestStep(app_context=context, default_wakeup_minutes=7, max_file_bytes=4)()
    agent.reply.assert_not_awaited()
    agent.append_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_scan_ignores_symlinks_and_rejects_old_statuses(runtime):
    context, agent = runtime
    outside = agent.workspace_path.parent / f"{agent.workspace_path.name}-outside.md"
    outside.write_text("outside")
    target = note(context)
    target.unlink()
    target.symlink_to(outside)
    assert (await push(context)).answer == {"sent": False}
    note(context, "interest_100000.md", status="ready")
    with pytest.raises(ValueError, match="Invalid interest status"):
        await push(context)
    assert outside.read_text() == "outside"


@pytest.mark.asyncio
@pytest.mark.parametrize("minutes", [0, -1, True, "5", None, float("inf"), float("nan")])
async def test_wakeup_rejects_invalid_delays(runtime, minutes):
    context, _ = runtime
    with pytest.raises(ValueError, match="positive finite"):
        await SetNextWakeupStep(app_context=context)(minutes=minutes)
    assert WAKEUP_KEY not in context.metadata


@pytest.mark.asyncio
async def test_loop_uses_configured_default_and_consumes_override_once(runtime, monkeypatch):
    context, _ = runtime
    context.metadata[WAKEUP_KEY] = 99
    stop, delays = asyncio.Event(), []
    step = PaLoopStep(app_context=context, default_wakeup_minutes=7)

    async def wait(waiter, *, timeout):
        waiter.close()
        delays.append(timeout)
        if len(delays) == 4:
            stop.set()
        else:
            raise asyncio.TimeoutError

    async def run(*_args, **kwargs):
        assert kwargs["stop_event"] is stop and kwargs["default_wakeup_minutes"] == 7
        if len(delays) == 1:
            await SetNextWakeupStep(app_context=context)(minutes=2)
        if len(delays) == 3:
            await SetNextWakeupStep(app_context=context)(minutes=20)
            return Response(success=False)
        return Response()

    step.run_job = AsyncMock(side_effect=run)
    monkeypatch.setattr("reme.steps.proactive.loop.asyncio.wait_for", wait)
    await step(stop_event=stop)
    assert delays == [420, 120, 420, 420]
    assert step.run_job.await_count == 3 and WAKEUP_KEY not in context.metadata


@pytest.mark.asyncio
async def test_background_close_interrupts_initial_wait(runtime):
    context, _ = runtime
    job = BackgroundJob(
        name="pa_loop",
        app_context=context,
        steps=[{"backend": "pa_loop_step", "default_wakeup_minutes": 7}],
    )
    context.jobs[job.name] = job
    await job.start()
    await asyncio.sleep(0)
    await asyncio.wait_for(job.close(), timeout=1)
    assert WAKEUP_KEY not in context.metadata


def test_proactive_jobs_register_new_tools_and_keep_them_private():
    jobs = resolve_app_config(config="proactive")["jobs"]
    assert jobs["pa_loop"]["backend"] == "background"
    for name in ("pa_proactive", "send_reminder", "set_next_wakeup"):
        assert not jobs[name]["enable_serve"]
    assert jobs["pa_loop"]["steps"][0]["default_wakeup_minutes"] == jobs["pa_proactive"]["default_wakeup_minutes"] == 5
    assert jobs["send_reminder"]["parameters"]["required"] == ["message", "files"]
