"""Focused contracts used by the local ReMe workspace web client."""

import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import ChunkEnum
from reme.schema import StreamChunk
from reme.steps.common.chat import ChatStep
from reme.steps.file_io.load import LoadStep
from reme.steps.file_io.save import SaveStep


class _StreamingAgent(BaseAgentWrapper):
    """Minimal agent that records resume/tool arguments and emits rich chunks."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reply_kwargs = {}

    async def reply(self, inputs, **kwargs) -> dict:
        raise AssertionError("ChatStep must always use reply_stream()")

    async def reply_stream(self, inputs, **kwargs):
        self.reply_kwargs = kwargs
        yield StreamChunk(chunk_type=ChunkEnum.TOOL_CALL, chunk="{}", tool_call_id="tool-1", tool_call_name="search")
        yield StreamChunk(chunk_type=ChunkEnum.CONTENT, chunk="hello", session_id="session-new")
        yield StreamChunk(chunk_type=ChunkEnum.REPLY_END, chunk="", session_id="session-new")


def test_save_step_preserves_complete_markdown(tmp_path):
    """The editor save endpoint writes frontmatter and body verbatim."""

    async def run():
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

    asyncio.run(run())


def test_load_step_returns_complete_content_and_mtime(tmp_path):
    """Editor loads are complete and include the concurrency token for save."""

    async def run():
        content = "# Large enough\n" + ("memory line\n" * 5000)
        target = tmp_path / "note.md"
        target.write_text(content, encoding="utf-8")

        response = await LoadStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(path="note.md")

        assert response.success is True
        assert response.answer == content
        assert response.metadata["size"] == target.stat().st_size
        assert response.metadata["mtime"]

    asyncio.run(run())


def test_load_step_rejects_file_larger_than_editor_limit(tmp_path):
    """Oversized files fail explicitly instead of returning editable truncation."""

    async def run():
        (tmp_path / "note.md").write_text("0123456789", encoding="utf-8")

        response = await LoadStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(
            path="note.md",
            max_bytes=5,
        )

        assert response.success is False
        assert response.metadata["code"] == "file_too_large"

    asyncio.run(run())


def test_save_step_rejects_external_change(tmp_path):
    """An obsolete mtime cannot overwrite a file changed by another editor."""

    async def run():
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

    asyncio.run(run())


def test_chat_step_streams_rich_chunks_and_resumes_session(tmp_path):
    """Web chat keeps StreamChunk metadata and exposes only read-only tools."""

    async def run():
        app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
        agent = _StreamingAgent(app_context=app_context)
        queue = asyncio.Queue()
        context = RuntimeContext(stream_queue=queue, query="hello", session_id="session-old")

        response = await ChatStep(agent_wrapper=agent, app_context=app_context)(context)

        chunks = [await queue.get(), await queue.get(), await queue.get()]
        assert response.answer == "hello"
        assert chunks[0].tool_call_name == "search"
        assert chunks[1].session_id == "session-new"
        assert chunks[2].metadata["answer"] == "hello"
        assert agent.reply_kwargs["resume"] == "session-old"
        assert agent.reply_kwargs["job_tools"] == ["search", "list", "stat", "traverse"]
        assert agent.reply_kwargs["builtin_tools"] == []
        system_prompt = agent.reply_kwargs["system_prompt"]
        assert f"Current date: {datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()}" in system_prompt
        assert f"Current working directory: {tmp_path.resolve()}" in system_prompt

    asyncio.run(run())


def test_chat_step_appends_environment_context_to_system_prompt_override(tmp_path):
    """A caller prompt override retains request-time date and cwd context."""

    async def run():
        app_context = ApplicationContext(workspace_dir=str(tmp_path), timezone="Asia/Shanghai")
        agent = _StreamingAgent(app_context=app_context)

        await ChatStep(agent_wrapper=agent, app_context=app_context)(
            query="hello",
            system_prompt="Custom prompt.",
            stream_queue=asyncio.Queue(),
        )

        system_prompt = agent.reply_kwargs["system_prompt"]
        assert system_prompt.startswith("Custom prompt.\n\n<environment_context>")
        assert "Current date:" in system_prompt
        assert f"Current working directory: {tmp_path.resolve()}" in system_prompt

    asyncio.run(run())
