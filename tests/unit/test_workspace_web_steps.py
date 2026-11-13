"""Focused contracts used by the local ReMe workspace web client."""

import asyncio
import os
from datetime import datetime

import pytest

from reme.components.agent_wrapper import CodexAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.job import StreamJob
from reme.components.runtime_context import RuntimeContext
from reme.config.config_parser import _load_config
from reme.enumeration import ChunkEnum
from reme.schema import StreamChunk
from reme.steps.common.chat import ChatStep
from reme.steps.file_io.load import LoadStep
from reme.steps.file_io.save import SaveStep


class _StreamingAgent(CodexAgentWrapper):
    """Minimal agent that records resume/tool arguments and emits rich chunks."""

    def __init__(self, **kwargs):
        super().__init__(app_server="", **kwargs)
        self.reply_kwargs = {}

    async def reply(self, inputs, **kwargs) -> dict:
        raise AssertionError("ChatStep must always use reply_stream()")

    async def create_session(self, **kwargs) -> str:
        return "session-new"

    async def reply_stream(self, inputs, **kwargs):
        self.reply_kwargs = kwargs
        yield StreamChunk(chunk_type=ChunkEnum.TOOL_CALL, chunk="{}", tool_call_id="tool-1", tool_call_name="search")
        yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="hello", session_id="session-new")
        yield StreamChunk(
            chunk_type=ChunkEnum.REPLY_END,
            event="end",
            status="completed",
            session_id="session-new",
            chunk={
                key: value
                for key, value in self._reply_result(session_id="session-new", result="hello", raw={}).items()
                if key != "events"
            },
        )


@pytest.mark.asyncio
async def test_save_step_preserves_complete_markdown(tmp_path):
    """The editor save endpoint writes frontmatter and body verbatim."""

    target = tmp_path / "daily" / "note.md"
    target.parent.mkdir(parents=True)
    target.write_text("old", encoding="utf-8")
    expected = datetime.fromtimestamp(target.stat().st_mtime).isoformat()
    content = "---\nname: note\ncustom: keep\n---\n\n# Updated\n"

    response = await SaveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(
        path="daily/note.md",
        content=content,
        expected_mtime=expected,
    )

    assert response.success is True
    assert target.read_text(encoding="utf-8") == content
    assert response.metadata["path"] == "daily/note.md"
    assert response.metadata["mtime"]


@pytest.mark.asyncio
async def test_load_step_returns_complete_content_and_mtime(tmp_path):
    """Editor loads are complete and include the concurrency token for save."""

    content = "# Large enough\n" + ("memory line\n" * 5000)
    target = tmp_path / "note.md"
    target.write_text(content, encoding="utf-8")

    response = await LoadStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(path="note.md")

    assert response.success is True
    assert response.answer == content
    assert response.metadata["size"] == target.stat().st_size
    assert response.metadata["mtime"]


@pytest.mark.asyncio
async def test_load_step_rejects_file_larger_than_editor_limit(tmp_path):
    """Oversized files fail explicitly instead of returning editable truncation."""

    (tmp_path / "note.md").write_text("0123456789", encoding="utf-8")

    response = await LoadStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(
        path="note.md",
        max_bytes=5,
    )

    assert response.success is False
    assert response.metadata["code"] == "file_too_large"


@pytest.mark.asyncio
async def test_save_step_rejects_external_change(tmp_path):
    """An obsolete mtime cannot overwrite a file changed by another editor."""

    target = tmp_path / "note.md"
    target.write_text("first", encoding="utf-8")
    expected = datetime.fromtimestamp(target.stat().st_mtime).isoformat()
    target.write_text("external", encoding="utf-8")
    newer = target.stat().st_mtime + 2
    os.utime(target, (newer, newer))

    response = await SaveStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(
        path="note.md",
        content="browser edit",
        expected_mtime=expected,
    )

    assert response.success is False
    assert response.metadata["code"] == "file_conflict"
    assert target.read_text(encoding="utf-8") == "external"


@pytest.mark.asyncio
async def test_chat_step_streams_rich_chunks_and_resumes_session(tmp_path):
    """Web chat keeps StreamChunk metadata and exposes only read-only tools."""

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _StreamingAgent(app_context=app_context)
    queue = asyncio.Queue()
    context = RuntimeContext(stream_queue=queue, query="hello", session_id="session-old")

    response = await ChatStep(agent_wrapper=agent, app_context=app_context)(context)

    chunks = [await queue.get(), await queue.get(), await queue.get()]
    assert response.answer == "hello"
    assert chunks[0].tool_call_name == "search"
    assert chunks[1].session_id == "session-new"
    assert chunks[2].chunk["result"] == "hello"
    assert agent.reply_kwargs["resume"] == "session-old"
    assert agent.reply_kwargs["job_tools"] == [
        "search",
        "list",
        "read",
        "read_image",
        "frontmatter_read",
        "stat",
        "traverse",
    ]
    assert agent.reply_kwargs["builtin_tools"] == []
    assert agent.reply_kwargs["system_prompt"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("system_prompt", ["", None, "Custom prompt.\n"])
async def test_chat_step_passes_system_prompt_override_verbatim(tmp_path, system_prompt):
    """A caller prompt reaches the wrapper without extra environment text."""

    app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
    agent = _StreamingAgent(app_context=app_context)

    await ChatStep(agent_wrapper=agent, app_context=app_context)(
        query="hello",
        system_prompt=system_prompt,
        stream_queue=asyncio.Queue(),
    )

    assert agent.reply_kwargs["system_prompt"] == (system_prompt or "")


@pytest.mark.asyncio
async def test_default_chat_uses_configured_prompt_and_allows_empty_override(tmp_path):
    """The built-in prompt comes from config and can be explicitly cleared."""
    config = _load_config("default")
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    agent = _StreamingAgent(app_context=app_context)
    app_context.components["agent_wrapper"] = {"default": agent}
    chat = StreamJob(name="chat", app_context=app_context, **config["jobs"]["chat"])

    async with chat:
        await chat(query="hello", stream_queue=asyncio.Queue())
        assert agent.reply_kwargs["system_prompt"] == config["jobs"]["chat"]["system_prompt"]
        assert agent.reply_kwargs["system_prompt"].startswith("You are ReMe Agent")
        await chat(query="hello", system_prompt="", stream_queue=asyncio.Queue())
        assert agent.reply_kwargs["system_prompt"] == ""


@pytest.mark.asyncio
async def test_proactive_chat_uses_native_tools_and_loads_its_prompt_template(tmp_path):
    """The PA stream uses configured Job tools and the localized prompt template."""

    from types import MethodType
    from unittest.mock import AsyncMock
    from .pa_helpers import bind_pa

    config = _load_config("proactive")
    app_context = ApplicationContext(workspace_dir=str(tmp_path))
    agent = CodexAgentWrapper(app_context=app_context, app_server="")
    agent.create_session = AsyncMock(return_value="session-new")
    agent.reply_stream = MethodType(_StreamingAgent.reply_stream, agent)
    bind_pa(app_context, agent)
    app_context.components["agent_wrapper"] = {"default": agent}
    chat = StreamJob(name="pa_chat", app_context=app_context, **config["jobs"]["pa_chat"])
    queue = asyncio.Queue()

    await chat.start()
    try:
        await chat(query="Help plan my day", stream_queue=queue)
        assert agent.reply_kwargs["job_tools"] == config["jobs"]["pa_chat"]["steps"][0]["job_tools"]
        assert agent.reply_kwargs["system_prompt"].startswith("## Role")
        memory = tmp_path / "daily/personal/MEMORY.md"
        assert str(memory) in agent.reply_kwargs["system_prompt"]
        assert f"Workspace: {tmp_path}" in agent.reply_kwargs["system_prompt"]
        assert "financial news" in memory.read_text(encoding="utf-8")
        assert agent.reply_kwargs["resume"] == "session-new"
        chunks = [await queue.get() for _ in range(5)]
        assert [chunk.chunk_type for chunk in chunks] == [
            ChunkEnum.DATA,
            ChunkEnum.TOOL_CALL,
            ChunkEnum.CONTENT,
            ChunkEnum.REPLY_END,
            ChunkEnum.DONE,
        ]
    finally:
        await chat.close()
