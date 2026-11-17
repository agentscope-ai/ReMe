"""Opt-in live checks for skills, additional instructions, and ReMe job tools."""

# pylint: disable=redefined-outer-name

import os
from pathlib import Path
import secrets

import pytest
import pytest_asyncio

from reme.application import Application
from reme.steps.base_step import BaseStep

pytestmark = pytest.mark.skipif(
    os.getenv("REME_CODEX_INTEGRATION") != "1",
    reason="set REME_CODEX_INTEGRATION=1 to run live Codex tests",
)


class _ProbeStep(BaseStep):
    """Record execution in the containing Application rather than a child process."""

    async def execute(self):
        """Capture the tool input and return a deterministic result."""
        text = self.context.data["text"]
        self.app_context.metadata.setdefault("codex_probe_calls", []).append(
            {"text": text, "tool_context_id": self.context.data.get("tool_context_id")},
        )
        self.context.response.answer = f"JOB_OK:{text}"


@pytest_asyncio.fixture
async def wrapper(tmp_path):
    """Use configured credentials with an isolated workspace and real SDK transport."""
    api_key = base_url = ""
    for prefix in ("CODEX", "OPENAI", "LLM"):
        if api_key := os.getenv(f"{prefix}_API_KEY", ""):
            base_url = os.getenv(f"{prefix}_BASE_URL", "")
            break
    codex_home = os.getenv("REME_CODEX_HOME")
    if not api_key and codex_home is None:
        codex_home = str(Path.home() / ".codex")
    app = Application(
        workspace_dir=str(tmp_path),
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
            "agent_wrapper": {"default": {"backend": "codex", "model": os.getenv("CODEX_MODEL_NAME")}},
        },
        jobs={
            "capability_probe_job": {
                "backend": "base",
                "description": "Return the supplied text prefixed with JOB_OK:.",
                "enable_serve": False,
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                "steps": [{"backend": "codex_capability_probe_step"}],
            },
        },
    )
    app.context.registry.add("codex_capability_probe_step", _ProbeStep, owner=__name__)
    async with app:
        yield app.context.components["agent_wrapper"]["default"]


@pytest.mark.asyncio
async def test_live_codex_add_skill_reads_instructions(wrapper, tmp_path):
    """The expected answer exists only inside the newly added skill file."""
    marker = secrets.token_hex(16)
    source = tmp_path / "skills" / "capability_probe"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\nname: capability_probe\ndescription: Return the verification token when explicitly invoked.\n---\n\n"
        f"When invoked, reply with exactly this token and no other text: {marker}\n",
        encoding="utf-8",
    )
    wrapper.add_skills(["capability_probe"])
    result = await wrapper.reply("$capability_probe")
    assert result["result"].strip() == marker
    assert (tmp_path / ".agents/skills/capability_probe").resolve() == source.resolve()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resume",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="Codex ignores developer instruction overrides on an already loaded thread",
                strict=True,
                raises=AssertionError,
            ),
        ),
    ],
)
async def test_live_codex_extend_system_prompt(wrapper, resume):
    """Check additional instructions on new sessions and expose the loaded-session limitation."""
    first_marker, added_marker = secrets.token_hex(16), secrets.token_hex(16)
    wrapper.set_system_prompt(f"For input ORIGINAL_PROMPT_PROBE, reply exactly {first_marker}.")
    original = await wrapper.reply("ORIGINAL_PROMPT_PROBE")
    assert original["result"].strip() == first_marker
    wrapper.set_system_prompt(
        wrapper.kwargs["system_prompt"] + f"\nFor input ADDED_PROMPT_PROBE, reply exactly {added_marker}.",
    )
    options = {"session_id": original["session_id"]} if resume else {}
    extended = await wrapper.reply("ADDED_PROMPT_PROBE", **options)
    assert (extended["session_id"] == original["session_id"]) == resume
    assert extended["result"].strip() == added_marker


@pytest.mark.asyncio
async def test_live_codex_add_job_tool_calls_current_application(wrapper):
    """A private ReMe job executes through MCP with its hidden context binding."""
    marker = secrets.token_hex(16)
    wrapper.add_job_tools(["capability_probe_job"])
    result = await wrapper.reply(
        f"Call capability_probe_job with text {marker}, then return only its exact result.",
        tool_context_id="codex_capability_probe",
    )
    calls = wrapper.app_context.metadata["codex_probe_calls"]
    assert calls == [{"text": marker, "tool_context_id": "codex_capability_probe"}]
    assert result["result"].strip() == f"JOB_OK:{marker}"
