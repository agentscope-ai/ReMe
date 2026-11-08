"""Application-owned proactive-agent sessions and query-only streaming."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import asyncio
from datetime import datetime
import json
from pathlib import Path
import tomllib
from unittest.mock import AsyncMock

import httpx
import pytest
import yaml

from reme.application import Application
from reme.components.agent_wrapper import AsAgentWrapper, BaseAgentWrapper, CcAgentWrapper, CodexAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.config import resolve_app_config
from reme.constants import DEFAULT_MAX_BYTES
from reme.enumeration import ChunkEnum
from reme.schema import SessionStatus, StreamChunk
from reme.steps.proactive.pa_chat import PaChatStep, PaNewSessionStep, PaSessionStep
from .pa_helpers import bind_pa


@pytest.mark.parametrize("override", [None, "custom-codex-model"])
def test_proactive_model_defaults_and_explicit_override(monkeypatch, override):
    if override is None:
        monkeypatch.delenv("CODEX_MODEL_NAME", raising=False)
    else:
        monkeypatch.setenv("CODEX_MODEL_NAME", override)
    config = resolve_app_config(config="proactive", log_config=False)
    agent = config["components"]["agent_wrapper"]["default"]
    server = config["components"]["codex_app_server"]["default"]
    model = server["model_catalog"]["models"][0]
    options = tomllib.loads("\n".join(server["config_overrides"]))
    selected = override or "qwen3.8-max-0902"
    assert agent["model"] == model["slug"] == selected
    assert agent["effort"] == model["default_reasoning_level"] == options["model_reasoning_effort"] == "xhigh"
    assert model["multi_agent_reasoning_effort"] == options["agents"]["default_subagent_reasoning_effort"] == "xhigh"
    assert [level["effort"] for level in model["supported_reasoning_levels"]] == ["xhigh"]
    assert model["input_modalities"] == ["text", "image"]
    assert model["context_window"] == options["model_context_window"] == 65536
    assert options["model_auto_compact_token_limit"] == 56000
    assert options["review_model"] == options["agents"]["default_subagent_model"] == selected
    assert options["memories"]["extract_model"] == options["memories"]["consolidation_model"] == selected
    assert options["features"]["memories"]
    assert options["memories"]["generate_memories"] and options["memories"]["use_memories"]
    if override is None:
        generic = resolve_app_config(config="default", log_config=False)
        assert generic["components"]["agent_wrapper"]["codex"]["model"] == "qwen3.8-max"


class PaAgent(CodexAgentWrapper):
    """Keep session creation separate from generation, with observable lifecycle gates."""

    def __init__(self, **kwargs):
        super().__init__(app_server="", **kwargs)
        self.created = []
        self.released = []
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def create_session(self, **kwargs):
        self.created.append(kwargs)
        return f"session-{len(self.created)}"

    async def get_session_status(self, session_id):
        return SessionStatus(session_id=session_id, status="idle")

    async def release_session(self, session_id):
        self.released.append(session_id)

    async def reply_stream(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        self.entered.set()
        await self.release.wait()
        session_id = kwargs.get("resume", "explicit-chat")
        yield StreamChunk(chunk_type=ChunkEnum.REPLY_START, session_id=session_id)
        yield StreamChunk(chunk="hello", session_id=session_id)
        yield StreamChunk(
            chunk_type=ChunkEnum.REPLY_END,
            event="end",
            status="completed",
            session_id=session_id,
            chunk={
                key: value
                for key, value in self._reply_result(session_id=session_id, result="hello", raw={}).items()
                if key != "events"
            },
        )


@pytest.fixture
def runtime(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path), language="zh")
    agent = PaAgent(app_context=context)
    bind_pa(context, agent)
    return context, agent


@pytest.mark.asyncio
async def test_current_session_is_read_only_and_new_session_does_not_generate(runtime):
    context, agent = runtime
    assert (await PaSessionStep(app_context=context)()).answer == {"session_id": None, "status": None}
    assert not agent.created
    memory = agent.workspace_path / "daily/personal/MEMORY.md"
    assert not memory.exists()
    step = PaNewSessionStep(app_context=context)
    assert (await step()).answer == "session-1"
    assert (await PaSessionStep(app_context=context)()).answer == {"session_id": "session-1", "status": "idle"}
    assert not agent.calls
    assert str(memory) in agent.created[0]["system_prompt"]
    assert agent.created[0]["job_tools"] == []
    assert memory.read_text(encoding="utf-8") == step.get_prompt("default_memory") + "\n"
    assert (await PaNewSessionStep(app_context=context)()).answer == "session-2"
    assert (await PaSessionStep(app_context=context)()).answer["session_id"] == "session-2"
    assert agent.released == ["session-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_prompt", [None, "", "配置的个人助手提示词。\n"])
async def test_chat_creates_once_and_resumes_across_fresh_steps(runtime, custom_prompt):
    context, agent = runtime
    prompt_dict = {} if custom_prompt is None else {"system_prompt_zh": custom_prompt}
    for query in ("first", "second"):
        queue = asyncio.Queue()
        response = await PaChatStep(app_context=context, prompt_dict=prompt_dict)(query=query, stream_queue=queue)
        assert response.success and response.answer == "hello"
        assert all(queue.get_nowait().session_id == "session-1" for _ in range(queue.qsize()))
    assert len(agent.created) == 1
    prompt = agent.created[0]["system_prompt"]
    if custom_prompt is None:
        assert prompt.startswith("## 角色")
        assert str(agent.workspace_path / "daily/personal/MEMORY.md") in prompt
    else:
        assert prompt == custom_prompt.strip()
    assert [kwargs["resume"] for _, kwargs in agent.calls] == ["session-1", "session-1"]
    assert agent.created[0] == {key: value for key, value in agent.calls[0][1].items() if key != "resume"}


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [None, "", "  ", 42])
async def test_invalid_query_does_not_create_a_session(runtime, query):
    context, agent = runtime
    with pytest.raises(ValueError, match="query"):
        await PaChatStep(app_context=context)(query=query, stream_queue=asyncio.Queue())
    assert not agent.created


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [PaChatStep, PaNewSessionStep, PaSessionStep])
async def test_request_cannot_override_the_session_or_server_configuration(runtime, step):
    context, agent = runtime
    with pytest.raises(ValueError, match="Unexpected PA parameters"):
        await step(app_context=context)(session_id="other", system_prompt="override")
    assert not agent.created


@pytest.mark.asyncio
async def test_busy_session_is_observable_and_rejects_chat_and_rotation(runtime):
    context, agent = runtime
    agent.release.clear()
    task = asyncio.create_task(PaChatStep(app_context=context)(query="hold", stream_queue=asyncio.Queue()))
    await agent.entered.wait()
    try:
        async with asyncio.timeout(1):
            assert (await PaSessionStep(app_context=context)()).answer == {
                "session_id": "session-1",
                "status": "active",
            }
            with pytest.raises(RuntimeError, match="busy"):
                await PaNewSessionStep(app_context=context)()
            with pytest.raises(RuntimeError, match="busy"):
                await PaChatStep(app_context=context)(query="other", stream_queue=asyncio.Queue())
        assert len(agent.created) == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert (await PaSessionStep(app_context=context)()).answer["status"] == "idle"
    assert (await PaNewSessionStep(app_context=context)()).answer == "session-2"


@pytest.mark.asyncio
async def test_creation_and_generation_failures_preserve_selection_and_release_lock(runtime, monkeypatch):
    context, agent = runtime
    await PaNewSessionStep(app_context=context)()
    monkeypatch.setattr(agent, "create_session", AsyncMock(side_effect=RuntimeError("creation failed")))
    with pytest.raises(RuntimeError, match="creation failed"):
        await PaNewSessionStep(app_context=context)()
    assert (await PaSessionStep(app_context=context)()).answer == {"session_id": "session-1", "status": "idle"}

    async def fail(*_args, **_kwargs):
        raise RuntimeError("generation failed")
        yield  # pylint: disable=unreachable

    monkeypatch.setattr(agent, "reply_stream", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        await PaChatStep(app_context=context)(query="hello", stream_queue=asyncio.Queue())
    assert (await PaSessionStep(app_context=context)()).answer == {"session_id": "session-1", "status": "idle"}


@pytest.mark.asyncio
@pytest.mark.parametrize("first_chat", [False, True])
async def test_cancelled_creation_finishes_publication_before_releasing_the_pa_lock(runtime, monkeypatch, first_chat):
    context, agent = runtime
    entered, release = asyncio.Event(), asyncio.Event()
    create = agent.create_session

    async def hold(**kwargs):
        entered.set()
        await release.wait()
        return await create(**kwargs)

    monkeypatch.setattr(agent, "create_session", hold)
    operation = (
        PaChatStep(app_context=context)(query="hello", stream_queue=asyncio.Queue())
        if first_chat
        else PaNewSessionStep(app_context=context)()
    )
    task = asyncio.create_task(operation)
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    try:
        assert not task.done()
        assert (await PaSessionStep(app_context=context)()).answer["status"] == "active"
        with pytest.raises(RuntimeError, match="busy"):
            await PaNewSessionStep(app_context=context)()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert (await PaSessionStep(app_context=context)()).answer == {"session_id": "session-1", "status": "idle"}
    assert not agent.calls


@pytest.mark.asyncio
async def test_failure_to_release_previous_session_does_not_undo_successful_creation(runtime, monkeypatch):
    context, agent = runtime
    await PaNewSessionStep(app_context=context)()
    monkeypatch.setattr(agent, "release_session", AsyncMock(side_effect=RuntimeError("cleanup failed")))
    assert (await PaNewSessionStep(app_context=context)()).answer == "session-2"
    assert (await PaSessionStep(app_context=context)()).answer == {"session_id": "session-2", "status": "idle"}


@pytest.mark.asyncio
async def test_unavailable_session_returns_an_error_state_so_the_client_can_rotate(runtime, monkeypatch):
    context, agent = runtime
    await PaNewSessionStep(app_context=context)()
    monkeypatch.setattr(agent, "get_session_status", AsyncMock(side_effect=RuntimeError("session lost")))
    response = await PaSessionStep(app_context=context)()
    assert response.success and response.answer == {"session_id": "session-1", "status": "error"}
    assert response.metadata["error"] == "session lost"
    assert (await PaNewSessionStep(app_context=context)()).answer == "session-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_error", [False, True])
async def test_status_does_not_publish_a_session_superseded_during_the_native_read(runtime, monkeypatch, stale_error):
    context, agent = runtime
    await PaNewSessionStep(app_context=context)()
    entered, release = asyncio.Event(), asyncio.Event()
    status = agent.get_session_status

    async def hold(session_id):
        if session_id == "session-1":
            entered.set()
            await release.wait()
            if stale_error:
                raise RuntimeError("old empty session was released")
        return await status(session_id)

    monkeypatch.setattr(agent, "get_session_status", hold)
    task = asyncio.create_task(PaSessionStep(app_context=context)())
    await entered.wait()
    try:
        await PaNewSessionStep(app_context=context)()
    finally:
        release.set()
    response = await task
    assert response.answer == {"session_id": "session-2", "status": "idle"}
    assert not response.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [PaChatStep, PaNewSessionStep, PaSessionStep])
@pytest.mark.parametrize("wrapper", [BaseAgentWrapper, AsAgentWrapper, CcAgentWrapper])
async def test_pa_rejects_non_codex_wrappers_before_accessing_state(tmp_path, step, wrapper):
    context = ApplicationContext(workspace_dir=str(tmp_path))
    context.components["agent_wrapper"] = {"default": wrapper(app_context=context)}
    with pytest.raises(NotImplementedError, match="CodexAgentWrapper only"):
        await step(app_context=context)(**({"query": "hello"} if step is PaChatStep else {}))
    assert "pa" not in context.metadata
    assert not (tmp_path / "personal").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["", " \n\t", "# Preferences\n- Keep {original} text.\n", "x" * (DEFAULT_MAX_BYTES + 1)],
)
@pytest.mark.parametrize("first_chat", [False, True])
async def test_session_uses_memory_path_without_injecting_or_modifying_existing_contents(runtime, content, first_chat):
    context, agent = runtime
    memory = agent.workspace_path / "daily/personal/MEMORY.md"
    memory.parent.mkdir(parents=True)
    memory.write_text(content, encoding="utf-8")
    agent._cwd = agent.workspace_path / "another-directory"
    if first_chat:
        await PaChatStep(app_context=context)(query="hello", stream_queue=asyncio.Queue())
    else:
        await PaNewSessionStep(app_context=context)()
        assert not agent.calls
    prompt = agent.created[0]["system_prompt"]
    assert str(memory) in prompt
    assert str(agent.cwd) not in prompt
    assert "修改前先读最新完整记忆" in prompt
    assert "主人要求遗忘时立即删除" in prompt
    assert "保留其他有效事实" in prompt
    if content.strip():
        assert content.strip() not in prompt
    assert memory.read_text(encoding="utf-8") == content


@pytest.mark.asyncio
async def test_chat_uses_current_prompt_and_tools_without_a_session_parameter_cache(runtime):
    context, agent = runtime
    await PaNewSessionStep(app_context=context, prompt_dict={"system_prompt_zh": "原来的提示词。"})()
    memory = agent.workspace_path / "daily/personal/MEMORY.md"
    memory.write_text("# Projects\n- Build ReMe.\n", encoding="utf-8")
    for _ in range(2):
        await PaChatStep(
            app_context=context,
            prompt_dict={"system_prompt_zh": "修改后的提示词。"},
            job_tools=["search"],
        )(
            query="continue",
            stream_queue=asyncio.Queue(),
        )
        assert agent.calls[-1][1] == {
            "resume": "session-1",
            "system_prompt": "修改后的提示词。",
            "job_tools": ["search"],
            "injected_job_kwargs": {"pa_caller": "chat"},
        }
        assert "reply_kwargs" not in vars(context.components["pa"]["default"])
    await PaNewSessionStep(app_context=context, prompt_dict={"system_prompt_zh": "修改后的提示词。"})()
    assert agent.created[1]["system_prompt"] == "修改后的提示词。"
    assert memory.read_text(encoding="utf-8") == "# Projects\n- Build ReMe.\n"


@pytest.mark.asyncio
async def test_personal_memory_follows_configured_daily_directory(runtime):
    context, agent = runtime
    context.app_config.daily_dir = "summaries"
    memory = agent.workspace_path / "summaries/personal/MEMORY.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("主人正在研究桌面交互。", encoding="utf-8")
    legacy = agent.workspace_path / "other/MEMORY.md"
    legacy.parent.mkdir()
    legacy.write_text("旧路径的内容不应加载。", encoding="utf-8")
    await PaNewSessionStep(app_context=context)()
    prompt = agent.created[0]["system_prompt"]
    assert str(memory) in prompt and str(legacy) not in prompt
    assert memory.read_text(encoding="utf-8") == "主人正在研究桌面交互。"
    assert legacy.read_text(encoding="utf-8") == "旧路径的内容不应加载。"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["directory", "escape", "permission"])
async def test_memory_initialization_errors_preserve_the_current_session(runtime, tmp_path, monkeypatch, failure):
    context, agent = runtime
    await PaNewSessionStep(app_context=context)()
    previous = context.components["pa"]["default"].session_id
    memory = tmp_path / "daily/personal/MEMORY.md"
    memory.unlink()
    if failure == "directory":
        memory.mkdir()
        error, message = ValueError, "regular file"
    elif failure == "escape":
        memory.symlink_to(tmp_path.parent / "outside-memory.md")
        error, message = ValueError, "inside the workspace"
    else:
        monkeypatch.setattr(
            "reme.steps.proactive.base_pa_step.write_file_safe",
            AsyncMock(side_effect=PermissionError("write denied")),
        )
        error, message = PermissionError, "write denied"
    with pytest.raises(error, match=message):
        await PaNewSessionStep(app_context=context)()
    assert context.components["pa"]["default"].session_id == previous
    assert len(agent.created) == 1 and not agent.released


@pytest.mark.asyncio
@pytest.mark.parametrize("app_language, step_language", [("", ""), ("en", ""), ("zh", ""), ("en", "zh"), ("zh", "en")])
@pytest.mark.parametrize("content", [None, '# Preferences\n{"project": "{name}", "style": "concise"}\n'])
async def test_pa_templates_follow_language_and_preserve_memory(tmp_path, app_language, step_language, content):
    context = ApplicationContext(workspace_dir=str(tmp_path), language=app_language)
    agent = PaAgent(app_context=context)
    bind_pa(context, agent)
    memory = tmp_path / "daily/personal/MEMORY.md"
    if content is not None:
        memory.parent.mkdir(parents=True)
        memory.write_text(content, encoding="utf-8")
    step = PaNewSessionStep(app_context=context, language=step_language)
    await step()
    prompt = agent.created[0]["system_prompt"]
    chinese = (step_language or app_language) == "zh"
    assert prompt.startswith("## 角色" if chinese else "## Role")
    assert ("对话 Agent" if chinese else "chat agent") in prompt
    assert f"{'工作区：' if chinese else 'Workspace: '}{tmp_path}" in prompt
    assert "【工作流】" not in prompt and "[Workflow]" not in prompt
    assert "会话创建时间" not in prompt and "Session created at" not in prompt
    assert "AGENTS.md" not in prompt
    assert str(memory.parent / "INTERESTS.md") not in prompt
    assert str(memory) in prompt
    if content is None:
        assert memory.read_text(encoding="utf-8") == step.get_prompt("default_memory") + "\n"
        assert ("财经新闻" if chinese else "financial news") in memory.read_text(encoding="utf-8")
        assert not (memory.parent / "INTERESTS.md").exists()
    else:
        assert content.strip() not in prompt
        assert memory.read_text(encoding="utf-8") == content


@pytest.mark.asyncio
async def test_pa_prompt_and_default_memory_overrides_only_seed_missing_files(runtime):
    context, agent = runtime
    prompts = {
        "system_prompt_zh": "自定义人设。\n文件：{memory_path}",
        "default_memory_zh": "# 自定义初始记忆\n保留 {原文}。",
    }
    await PaNewSessionStep(app_context=context, prompt_dict=prompts)()
    memory = agent.workspace_path / "daily/personal/MEMORY.md"
    assert agent.created[0]["system_prompt"] == f"自定义人设。\n文件：{memory}"
    assert memory.read_text(encoding="utf-8") == prompts["default_memory_zh"] + "\n"
    memory.write_text("更新后的 {记忆}。", encoding="utf-8")
    prompts["default_memory_zh"] = "不应替换已有记忆。"
    await PaNewSessionStep(app_context=context, prompt_dict=prompts)()
    assert memory.read_text(encoding="utf-8") == "更新后的 {记忆}。"


@pytest.mark.asyncio
@pytest.mark.parametrize("config_name", ["default", "proactive"])
async def test_http_chat_modes_follow_the_selected_configuration(tmp_path, config_name, monkeypatch):
    monkeypatch.setattr(
        "reme.steps.proactive.base_pa_step.BasePaStep._now",
        lambda _self: datetime.fromisoformat("2026-11-08T12:00:00+08:00"),
    )
    config = resolve_app_config(config=config_name, log_config=False)
    app = Application(
        workspace_dir=str(tmp_path),
        enable_logo=False,
        log_to_file=False,
        log_to_console=False,
        language=config.get("language", ""),
        service={"backend": "http", "web_enabled": False, "mcp_enabled": False},
        jobs={
            name: config["jobs"][name]
            for name in ("chat", "pa_chat", "pa_session", "pa_new_session")
            if name in config["jobs"]
        },
    )
    agent = PaAgent(app_context=app.context)
    app.context.components["agent_wrapper"] = {"default": agent}
    if config_name == "proactive":
        bind_pa(app.context, agent)
    service = app.context.service
    service.build_service(app)
    service.add_jobs(app)
    async with app:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.service),
            base_url="http://test",
        ) as client:
            if config_name == "default":
                assert "chat" in config["jobs"]
                for name in ("pa_chat", "pa_session", "pa_new_session"):
                    assert name not in config["jobs"]
                    assert (await client.post(f"/{name}", json={})).status_code == 404
                response = await client.post("/chat", json={"query": "hello", "session_id": "explicit-session"})
                assert response.headers["content-type"].startswith("text/event-stream")
                assert "data:[DONE]" in response.text
                assert agent.calls[-1][1]["resume"] == "explicit-session"
                assert not agent.created and "pa" not in app.context.metadata
                return
            assert "chat" not in config["jobs"]
            model = config["components"]["codex_app_server"]["default"]["model_catalog"]["models"][0]
            assert model["model_messages"]["instructions_template"] == "" and "base_instructions" not in model
            assert "system_prompt" not in config["jobs"]["pa_chat"]["steps"][0]
            assert config["language"] == "zh"
            assert (await client.post("/chat", json={"query": "hello"})).status_code == 404
            current = (await client.post("/pa_session", json={})).json()
            assert current["success"] and current["answer"] == {"session_id": None, "status": None}
            new = (await client.post("/pa_new_session", json={})).json()
            assert new["success"] and new["answer"] == "session-1" and not agent.calls
            response = await client.post("/pa_chat", json={"query": "hello"})
            assert response.headers["content-type"].startswith("text/event-stream")
            chunks = [json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:{")]
            assert "data:[DONE]" in response.text and chunks[-1]["chunk"]["session_id"] == "session-1"
            assert agent.created[0] == {key: value for key, value in agent.calls[0][1].items() if key != "resume"}
            forbidden = (await client.post("/pa_new_session", json={"session_id": "other"})).json()
            assert not forbidden["success"]
            assert (await client.post("/pa_session", json={})).json()["answer"]["session_id"] == "session-1"


@pytest.mark.parametrize(
    "module,memory_field",
    [("pa_chat", "memory_path"), ("proactive_agent", "memory_path"), ("cron_agent", "memory")],
)
def test_role_prompts_have_matching_localized_placeholders(module, memory_field):
    from string import Formatter

    root = Path(__file__).resolve().parents[2]
    prompts = yaml.safe_load((root / f"reme/steps/proactive/{module}.yaml").read_text())

    def fields(value):
        return {field for _, field, _, _ in Formatter().parse(value) if field}

    assert fields(prompts["system_prompt"]) == fields(prompts["system_prompt_zh"])
    assert memory_field in fields(prompts["system_prompt"])
