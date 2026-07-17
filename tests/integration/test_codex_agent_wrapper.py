"""Opt-in live coverage for the Codex app-server wrapper.

Run with ``REME_CODEX_INTEGRATION=1``. These tests consume the caller's active
API-key or Codex OAuth account and are intentionally excluded from normal CI.
"""

# pylint: disable=protected-access

import asyncio
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from reme.components.agent_wrapper.codex_agent_wrapper import CodexAgentWrapper
from reme.enumeration import ChunkEnum, ComponentEnum
from reme.schema import ApplicationConfig, Response

pytestmark = pytest.mark.skipif(
    os.getenv("REME_CODEX_INTEGRATION") != "1",
    reason="set REME_CODEX_INTEGRATION=1 to run live Codex tests",
)


class _StructuredResult(BaseModel):
    marker: str


class _CustomJob:
    name = "only_custom"
    description = "Return the fixed marker CUSTOM_JOB_OK."
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    async def __call__(self, **_kwargs):
        return Response(answer="CUSTOM_JOB_OK")


class _DraftJob:
    def __init__(self, name: str, parameters: dict):
        self.name = name
        self.description = f"Live tool-context contract job: {name}"
        self.parameters = parameters


def _child_pids(root_pid: int) -> set[int]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid="],
        check=True,
        capture_output=True,
        text=True,
    )
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        pid, parent = (int(value) for value in line.split())
        children.setdefault(parent, []).append(pid)
    found: set[int] = set()
    pending = list(children.get(root_pid, []))
    while pending:
        pid = pending.pop()
        if pid not in found:
            found.add(pid)
            pending.extend(children.get(pid, []))
    return found


@pytest.mark.asyncio
async def test_live_codex_reply_stream_tools_skills_resume_fork_approval_and_close(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project_skill = Path(__file__).resolve().parents[2] / "skills" / "reme_memory"
    skills_root = workspace / "skills"
    skills_root.mkdir()
    (skills_root / "reme_memory").symlink_to(project_skill, target_is_directory=True)

    app_config = ApplicationConfig(
        workspace_dir=str(workspace),
        mem_session_dir="sessions",
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "mcp"},
        jobs={
            "only_custom": {
                "backend": "base",
                "description": _CustomJob.description,
                "parameters": _CustomJob.parameters,
                "steps": [],
            },
            "add_live_draft": {
                "backend": "base",
                "description": "Append text to the current tool context.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                "steps": [{"backend": "add_draft_step"}],
            },
            "read_live_draft": {
                "backend": "base",
                "description": "Read text accumulated in the current tool context.",
                "parameters": {"type": "object", "properties": {}},
                "steps": [{"backend": "read_all_draft_step"}],
            },
        },
        components={ComponentEnum.AS_LLM: {}},
    )
    context = SimpleNamespace(
        app_config=app_config,
        jobs={
            "only_custom": _CustomJob(),
            "add_live_draft": _DraftJob(
                "add_live_draft",
                {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            "read_live_draft": _DraftJob(
                "read_live_draft",
                {"type": "object", "properties": {}},
            ),
        },
    )
    codex_home = os.getenv("REME_CODEX_HOME")
    if codex_home is None and not any(os.getenv(name) for name in ("CODEX_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")):
        codex_home = str(Path.home() / ".codex")
    wrapper = CodexAgentWrapper(app_context=context, codex_home=codex_home)

    await wrapper.start()
    first = await wrapper.reply(
        "Call add_live_draft with text STATE_OK, then reply FIRST_TURN_OK.",
        job_tools=["add_live_draft", "read_live_draft"],
        tool_context_id="live-context",
    )
    assert "FIRST_TURN_OK" in first["last_message"]

    resumed = await wrapper.reply(
        "Call read_live_draft. Include its exact result and RESUME_OK in your answer.",
        resume=first["session_id"],
        job_tools=["add_live_draft", "read_live_draft"],
        tool_context_id="live-context",
    )
    assert resumed["session_id"] == first["session_id"]
    assert "RESUME_OK" in resumed["last_message"]
    assert "STATE_OK" in resumed["last_message"]

    isolated = await wrapper.reply(
        "Call read_live_draft. If it is empty, reply ISOLATED_OK.",
        job_tools=["add_live_draft", "read_live_draft"],
        tool_context_id="other-context",
    )
    assert "ISOLATED_OK" in isolated["last_message"]
    assert "STATE_OK" not in isolated["last_message"]

    resumed_after_isolated_context = await wrapper.reply(
        "Call read_live_draft again and include its exact result.",
        resume=first["session_id"],
        job_tools=["add_live_draft", "read_live_draft"],
        tool_context_id="live-context",
    )
    assert "STATE_OK" in resumed_after_isolated_context["last_message"]

    forked = await wrapper.reply(
        "Reply with exactly FORK_OK.",
        resume=first["session_id"],
        fork_session=True,
        tool_context_id="fork-context",
    )
    assert forked["session_id"] != first["session_id"]
    assert "FORK_OK" in forked["last_message"]

    structured = await wrapper.reply(
        "Return marker STRUCTURED_OK.",
        output_schema=_StructuredResult,
    )
    assert structured["structured_output"] == {"marker": "STRUCTURED_OK"}

    tool_result = await wrapper.reply(
        "Call the only_custom tool once, then include its result in your answer.",
        job_tools=["only_custom"],
        tool_context_id="tool-context",
    )
    assert "CUSTOM_JOB_OK" in tool_result["last_message"]

    skill_result = await wrapper.reply(
        "Use the reme_memory skill. Read only its instructions and reply SKILL_OK; do not run its scripts or CLI.",
        skills=["reme_memory"],
    )
    assert "SKILL_OK" in skill_result["last_message"]
    assert (workspace / ".agents" / "skills" / "reme_memory").is_symlink()

    outside_target = tmp_path / "approval-target.txt"
    approval_chunks = [
        chunk
        async for chunk in wrapper.reply_stream(
            f"Try to write the word approved to {outside_target} using a shell command.",
            approval_mode="auto_review",
            sandbox="workspace-write",
        )
    ]
    assert approval_chunks[0].chunk_type == ChunkEnum.REPLY_START
    assert approval_chunks[-1].chunk_type == ChunkEnum.REPLY_END
    assert any(chunk.chunk_type == ChunkEnum.APPROVAL for chunk in approval_chunks)

    codex_proc = wrapper._codex._client._sync._proc
    assert codex_proc is not None and codex_proc.poll() is None
    child_pids = _child_pids(codex_proc.pid)
    await wrapper.close()
    await asyncio.sleep(0.2)

    assert codex_proc.poll() is not None
    running_pids = {
        int(line)
        for line in subprocess.run(
            ["ps", "-axo", "pid="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    }
    assert child_pids.isdisjoint(running_pids)
    assert wrapper._codex is None
    assert wrapper._mcp_snapshot_path is None

    second_context = SimpleNamespace(app_config=app_config, jobs=context.jobs.copy())
    second_wrapper = CodexAgentWrapper(app_context=second_context, codex_home=codex_home)
    await second_wrapper.start()
    isolated_application = await second_wrapper.reply(
        "Call read_live_draft. If it is empty, reply NEW_APPLICATION_OK.",
        job_tools=["read_live_draft"],
        tool_context_id="live-context",
    )
    await second_wrapper.close()
    assert "NEW_APPLICATION_OK" in isolated_application["last_message"]
    assert "STATE_OK" not in isolated_application["last_message"]
