"""Opt-in live Codex SDK coverage; set REME_CODEX_INTEGRATION=1 to use credentials."""

# pylint: disable=protected-access

import asyncio
import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from reme.application import Application
from reme.enumeration import ChunkEnum
from reme.steps.base_step import BaseStep

pytestmark = pytest.mark.skipif(
    os.getenv("REME_CODEX_INTEGRATION") != "1",
    reason="set REME_CODEX_INTEGRATION=1 to run live Codex tests",
)


class _StructuredResult(BaseModel):
    marker: str


class _CustomStep(BaseStep):
    """Exercise a parent-only registration and the actual parent metadata."""

    async def execute(self):
        self.app_context.metadata["live_custom_calls"] = self.app_context.metadata.get("live_custom_calls", 0) + 1
        self.context.response.answer = "CUSTOM_JOB_OK"


@pytest.mark.asyncio
async def test_live_shared_codex_sessions_tools_streaming_and_close(tmp_path):
    """Use one real SDK process and execute every job inside the containing Application."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skills_root = workspace / "skills"
    skills_root.mkdir()
    project_skill = Path(__file__).resolve().parents[2] / "skills" / "reme_memory"
    (skills_root / "reme_memory").symlink_to(project_skill, target_is_directory=True)
    api_key = base_url = ""
    for prefix in ("CODEX", "OPENAI", "LLM"):
        if api_key := os.getenv(f"{prefix}_API_KEY", ""):
            base_url = os.getenv(f"{prefix}_BASE_URL", "")
            break
    codex_home = os.getenv("REME_CODEX_HOME")
    if not api_key and codex_home is None:
        codex_home = str(Path.home() / ".codex")
    app = Application(
        workspace_dir=str(workspace),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "mcp"},
        components={
            "codex_app_server": {
                "default": {
                    "backend": "sdk",
                    "auth_mode": "api_key" if api_key else "oauth",
                    "api_key": api_key,
                    "base_url": base_url,
                    "codex_home": codex_home,
                },
            },
            "agent_wrapper": {
                name: {"backend": "codex", "model": os.getenv("CODEX_MODEL_NAME")} for name in ("first", "second")
            },
        },
        jobs={
            "only_custom": {
                "backend": "base",
                "description": "Return CUSTOM_JOB_OK.",
                "enable_serve": False,
                "steps": [{"backend": "live_custom_step"}],
            },
            "add_draft": {
                "backend": "base",
                "description": "Append text to the current tool context.",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                "steps": [{"backend": "add_draft_step"}],
            },
            "read_draft": {
                "backend": "base",
                "description": "Read text accumulated in the current tool context.",
                "parameters": {"type": "object", "properties": {}},
                "steps": [{"backend": "read_all_draft_step"}],
            },
        },
    )
    app.context.registry.add("live_custom_step", _CustomStep, owner=__name__)
    server = app.context.components["codex_app_server"]["default"]
    async with app:
        first, second = (app.context.components["agent_wrapper"][name] for name in ("first", "second"))
        original = await first.reply(
            "Call add_draft with text STATE_OK, then reply FIRST_OK.",
            job_tools=["add_draft", "read_draft"],
            tool_context_id="one",
        )
        assert "FIRST_OK" in original["last_message"]
        resumed = await second.reply(
            "Call read_draft and include its exact result.",
            session_id=original["session_id"],
            job_tools=["add_draft", "read_draft"],
            tool_context_id="one",
        )
        assert resumed["session_id"] == original["session_id"]
        assert "STATE_OK" in resumed["last_message"]
        isolated = await second.reply(
            "Call read_draft. If it is empty, reply ISOLATED_OK.",
            job_tools=["read_draft"],
            tool_context_id="two",
        )
        assert isolated["session_id"] != original["session_id"]
        assert "ISOLATED_OK" in isolated["last_message"]
        assert "STATE_OK" not in isolated["last_message"]
        forked = await first.reply("Reply FORK_OK.", resume=original["session_id"], fork_session=True)
        assert forked["session_id"] != original["session_id"]
        assert "FORK_OK" in forked["last_message"]
        custom = await first.reply("Call only_custom and include its result.", job_tools=["only_custom"])
        assert "CUSTOM_JOB_OK" in custom["last_message"]
        assert app.context.metadata["live_custom_calls"] >= 1
        structured = await first.reply("Return marker STRUCTURED_OK.", output_schema=_StructuredResult)
        assert structured["structured_output"] == {"marker": "STRUCTURED_OK"}
        await first.compact_session(structured["session_id"])
        skill = await first.reply(
            "Read the reme_memory skill instructions and reply SKILL_OK. Do not execute its scripts.",
            skills=["reme_memory"],
        )
        assert "SKILL_OK" in skill["last_message"]
        assert (workspace / ".agents/skills/reme_memory").is_symlink()
        concurrent = await asyncio.gather(first.reply("Reply ONE."), second.reply("Reply TWO."))
        assert concurrent[0]["session_id"] != concurrent[1]["session_id"]
        chunks = [chunk async for chunk in second.reply_stream("Reply STREAM_OK.")]
        assert chunks[0].chunk_type == ChunkEnum.REPLY_START
        assert chunks[-1].chunk_type == ChunkEnum.REPLY_END
        process = server._codex._client._sync._proc
        assert process.poll() is None
        await first.close()
        assert process.poll() is None
        assert (await second.reply("Reply STILL_OK."))["result"]
    assert process.poll() is not None
    assert server._codex is None
    assert server.jobs_mcp._task is None
