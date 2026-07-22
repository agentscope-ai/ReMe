"""Tests for the Claude Code agent wrapper."""

from dataclasses import replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper
from reme.components.agent_wrapper.cc_session_store import CcFileSessionStore
from reme.components.application_context import ApplicationContext
from reme.enumeration import ChunkEnum, ComponentEnum

# pylint: disable=protected-access


def _wrapper(tmp_path: Path) -> CcAgentWrapper:
    return CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))


def _skill_roots(tmp_path: Path) -> tuple[Path, Path]:
    return (
        tmp_path / ".claude" / "skills",
        tmp_path / "mem_session" / "claude_config" / "skills",
    )


def test_ensure_claude_skill_dir_adds_selected_skills_without_replacing_existing(
    tmp_path,
):
    """Selected workspace skills are added while unrelated Claude skills remain."""
    project_skills = tmp_path / "skills"
    (project_skills / "one").mkdir(parents=True)
    (project_skills / "two").mkdir()
    for name in ("one", "two"):
        (project_skills / name / "SKILL.md").write_text(f"# {name}", encoding="utf-8")
    config_dir = tmp_path / "mem_session" / "claude_config"

    for root in _skill_roots(tmp_path):
        existing = root / "existing"
        existing.mkdir(parents=True)
        (existing / "SKILL.md").write_text("existing", encoding="utf-8")

    _wrapper(tmp_path)._ensure_claude_skill_dir(config_dir, ["one"])

    for root in _skill_roots(tmp_path):
        assert (root / "one").is_symlink()
        assert (root / "one").resolve() == (project_skills / "one").resolve()
        assert not (root / "two").exists()
        assert (root / "existing" / "SKILL.md").read_text(encoding="utf-8") == "existing"


def test_ensure_claude_skill_dir_all_adds_each_project_skill(tmp_path):
    """The all selector creates child links instead of replacing the skills root."""
    project_skills = tmp_path / "skills"
    (project_skills / "one").mkdir(parents=True)
    (project_skills / "two").mkdir()
    for name in ("one", "two"):
        (project_skills / name / "SKILL.md").write_text(f"# {name}", encoding="utf-8")
    config_dir = tmp_path / "mem_session" / "claude_config"

    _wrapper(tmp_path)._ensure_claude_skill_dir(config_dir, "all")

    for root in _skill_roots(tmp_path):
        assert root.is_dir()
        assert not root.is_symlink()
        assert {path.name for path in root.iterdir()} == {"one", "two"}


def test_ensure_claude_skill_dir_preserves_existing_directory_link(tmp_path):
    """An existing skills link is user-owned and remains untouched."""
    project_skills = tmp_path / "skills"
    (project_skills / "one").mkdir(parents=True)
    (project_skills / "one" / "SKILL.md").write_text("# one", encoding="utf-8")
    config_dir = tmp_path / "mem_session" / "claude_config"
    legacy_root = tmp_path / ".claude" / "skills"
    legacy_root.parent.mkdir(parents=True)
    legacy_root.symlink_to(project_skills, target_is_directory=True)

    _wrapper(tmp_path)._ensure_claude_skill_dir(config_dir, ["one"])

    assert legacy_root.is_dir()
    assert legacy_root.is_symlink()
    assert (legacy_root / "one").resolve() == (project_skills / "one").resolve()


@pytest.mark.asyncio
async def test_file_session_store_conforms_to_latest_sdk(tmp_path):
    """The file store follows the SDK project/session/subkey contract."""
    from claude_agent_sdk.testing import run_session_store_conformance

    sequence = count()
    await run_session_store_conformance(lambda: CcFileSessionStore(tmp_path / str(next(sequence))))


def test_ensure_claude_skill_dir_rejects_paths_as_skill_names(tmp_path):
    """Skill selectors cannot escape the project skills directory."""
    (tmp_path / "skills").mkdir()

    with pytest.raises(ValueError, match="Invalid skill name"):
        _wrapper(tmp_path)._ensure_claude_skill_dir(tmp_path / "config", ["../outside"])


def test_configured_skills_use_latest_sdk_allowlist(tmp_path):
    """Selected ReMe skills are passed through using the SDK's list semantics."""
    project_skills = tmp_path / "skills"
    (project_skills / "one").mkdir(parents=True)
    (project_skills / "two").mkdir()
    for name in ("one", "two"):
        (project_skills / name / "SKILL.md").write_text(f"# {name}", encoding="utf-8")

    opts = _wrapper(tmp_path)._build_options("hello", skills=["one"])

    assert opts.skills == ["one"]
    for root in _skill_roots(tmp_path):
        assert (root / "one").resolve() == (project_skills / "one").resolve()
        assert not (root / "two").exists()


def test_configured_project_path_sources_skills_outside_workspace(tmp_path):
    """Claude Code links project skills while keeping sessions in the workspace."""
    project = tmp_path / "project"
    workspace = project / ".reme"
    skill = project / "skills" / "serper-search"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Serper Search", encoding="utf-8")
    wrapper = CcAgentWrapper(
        app_context=ApplicationContext(workspace_dir=str(workspace)),
        project_path="..",
    )

    opts = wrapper._build_options("hello", skills=["serper-search"])

    assert opts.cwd == project
    assert (project / ".claude" / "skills" / "serper-search").resolve() == skill
    assert (workspace / "mem_session" / "claude_config" / "skills" / "serper-search").resolve() == skill


def test_sdk_native_system_prompt_preset_is_preserved(tmp_path):
    """System prompt dictionaries pass directly to the latest SDK."""
    opts = _wrapper(tmp_path)._build_options(
        "hello",
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": "custom prompt",
        },
    )

    assert opts.system_prompt == {
        "type": "preset",
        "preset": "claude_code",
        "append": "custom prompt",
    }


def test_web_search_is_disallowed_by_default(tmp_path):
    """Claude Code keeps its normal tools except for web search."""
    opts = _wrapper(tmp_path)._build_options("hello")

    assert opts.disallowed_tools == ["WebSearch"]


def test_api_credentials_use_only_wrapper_config(tmp_path, monkeypatch):
    """Claude Code credentials do not fall back to ambient or shared LLM configuration."""
    wrapper = _wrapper(tmp_path)
    wrapper.app_context.app_config.environment = {
        "ANTHROPIC_AUTH_TOKEN": "application-key",
        "ANTHROPIC_BASE_URL": "https://application.example.test",
        "TOOL_ENV": "preserved",
    }
    wrapper.app_context.app_config.components[ComponentEnum.AS_LLM] = {
        "default": SimpleNamespace(
            credential={
                "api_key": "default-key",
                "base_url": "https://default.example.test",
            },
        ),
    }
    for name in ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "ambient-key")
    for name in ("ANTHROPIC_BASE_URL", "CLAUDE_CODE_BASE_URL", "LLM_BASE_URL"):
        monkeypatch.setenv(name, "https://ambient.example.test")
    configured = wrapper._build_options(  # pylint: disable=protected-access
        "hello",
        api_key="configured-key",
        base_url="https://configured.example.test",
        credential={"api_key": "nested-key", "base_url": "https://nested.example.test"},
    )
    assert configured.env["ANTHROPIC_AUTH_TOKEN"] == "configured-key"
    assert configured.env["ANTHROPIC_BASE_URL"] == "https://configured.example.test"
    assert configured.env["TOOL_ENV"] == "preserved"

    empty = wrapper._build_options(  # pylint: disable=protected-access
        "hello",
        credential={"api_key": "nested-key", "base_url": "https://nested.example.test"},
    )
    assert empty.env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert empty.env["ANTHROPIC_BASE_URL"] == ""


def test_build_options_accepts_empty_output_schema(tmp_path):
    """An empty schema remains a valid structured-output request."""
    opts = _wrapper(tmp_path)._build_options("hello", output_schema={})

    assert opts.output_format == {"type": "json_schema", "schema": {}}


def test_build_options_uses_native_sessions_and_allows_file_checkpointing(tmp_path):
    """Local Claude transcripts remain the default and do not conflict with checkpoints."""
    opts = _wrapper(tmp_path)._build_options("hello", enable_file_checkpointing=True)

    assert opts.enable_file_checkpointing is True
    assert opts.session_store is None
    assert opts.env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "mem_session" / "claude_config")


def test_build_options_preserves_explicit_session_store(tmp_path):
    """Callers can still opt into the SDK's external transcript mirror."""
    store = CcFileSessionStore(tmp_path / "mirror")

    opts = _wrapper(tmp_path)._build_options("hello", session_store=store)

    assert opts.session_store is store


def test_job_tools_reject_non_mapping_mcp_config_instead_of_discarding_it(tmp_path, monkeypatch):
    """Adding ReMe tools never silently replaces an SDK MCP config path."""
    wrapper = _wrapper(tmp_path)
    job = SimpleNamespace(name="remember", description="Remember", parameters={})
    monkeypatch.setattr(wrapper, "_resolve_job_tools", lambda _names: [job])

    with pytest.raises(ValueError, match="mcp_servers to be a mapping"):
        wrapper._build_options("hello", job_tools=["remember"], mcp_servers=tmp_path / "mcp.json")


def test_job_tools_merge_with_mapping_mcp_config(tmp_path, monkeypatch):
    """Existing MCP configuration remains reusable beside ReMe tools."""
    wrapper = _wrapper(tmp_path)
    job = SimpleNamespace(name="remember", description="Remember", parameters={})
    monkeypatch.setattr(wrapper, "_resolve_job_tools", lambda _names: [job])
    external = {"type": "http", "url": "https://mcp.example.test"}
    mcp_servers = {"external": external}
    allowed_tools = ["Read"]

    first = wrapper._build_options(
        "hello",
        job_tools=["remember"],
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
    )
    second = wrapper._build_options(
        "hello",
        job_tools=["remember"],
        mcp_servers=mcp_servers,
        allowed_tools=allowed_tools,
    )

    assert mcp_servers == {"external": external}
    assert allowed_tools == ["Read"]
    for opts in (first, second):
        assert opts.mcp_servers["external"] is external
        assert opts.mcp_servers[wrapper.MCP_SERVER_NAME]["type"] == "sdk"
        assert opts.allowed_tools == ["Read", "remember"]


@pytest.mark.asyncio
async def test_reply_preserves_falsy_structured_output(tmp_path, monkeypatch):
    """Falsy structured output is returned instead of being discarded."""
    from claude_agent_sdk import ResultMessage

    message = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="session-1",
        result="{}",
        structured_output={"placeholder": True},
    )

    async def query(**_kwargs):
        yield replace(message, structured_output={})

    monkeypatch.setattr("claude_agent_sdk.query", query)

    result = await _wrapper(tmp_path).reply("hello", output_schema={})

    assert "structured_output" in result
    assert result["structured_output"] == {}


def test_error_result_with_success_subtype_is_not_suppressed():
    """Latest SDK can report API failures with subtype=success and is_error=True."""
    from claude_agent_sdk import ResultMessage

    message = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=True,
        num_turns=1,
        session_id="session-1",
        errors=["upstream unavailable"],
        api_error_status=529,
    )

    chunks = CcAgentWrapper._result_message_to_chunks(message)

    error = next(chunk for chunk in chunks if chunk.chunk_type.value == "error")
    assert error.chunk == "upstream unavailable"
    assert error.metadata == {"api_error_status": 529}


def test_latest_sdk_server_tool_blocks_are_converted():
    """Server-side tools use the same unified call/result lifecycle."""
    from claude_agent_sdk import AssistantMessage, ServerToolResultBlock

    call = CcAgentWrapper._raw_event_to_chunk(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "tool-1",
                "name": "web_search",
            },
        },
    )
    message = AssistantMessage(
        content=[ServerToolResultBlock(tool_use_id="tool-1", content={"type": "web_search_result"})],
        model="claude",
    )
    results = CcAgentWrapper._message_content_to_chunks(message, visible_tool_call_ids={"tool-1"})

    assert call is not None and call.chunk_type.value == "tool_call"
    assert len(results) == 1
    assert results[0].chunk_type.value == "tool_result"
    assert results[0].chunk == {
        "tool_use_id": "tool-1",
        "content": {"type": "web_search_result"},
    }


@pytest.mark.asyncio
async def test_reply_stream_emits_one_reply_end_for_normal_sdk_lifecycle(tmp_path, monkeypatch):
    """Message delta reports usage while message stop is the sole lifecycle end."""
    from claude_agent_sdk import ResultMessage, StreamEvent

    async def query(**_kwargs):
        for event in (
            {
                "type": "message_start",
                "message": {"id": "message-1", "role": "assistant"},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ):
            yield StreamEvent(uuid="event-1", session_id="session-1", event=event)
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session-1",
            usage={"input_tokens": 1, "output_tokens": 2},
        )

    monkeypatch.setattr("claude_agent_sdk.query", query)
    chunks = [chunk async for chunk in _wrapper(tmp_path).reply_stream("hello")]

    assert sum(chunk.chunk_type == ChunkEnum.REPLY_END for chunk in chunks) == 1
    delta_usage = next(chunk for chunk in chunks if chunk.metadata.get("stop_reason") == "end_turn")
    assert delta_usage.chunk_type == ChunkEnum.USAGE
    assert delta_usage.output_tokens == 2


@pytest.mark.asyncio
async def test_reply_stream_only_reports_rejected_rate_limit(tmp_path, monkeypatch):
    """Rate-limit warnings are informational; only rejected is an error."""
    from claude_agent_sdk import RateLimitEvent, RateLimitInfo, ResultMessage

    async def query(**_kwargs):
        yield RateLimitEvent(
            rate_limit_info=RateLimitInfo(status="allowed_warning"),
            uuid="warning",
            session_id="session-1",
        )
        yield RateLimitEvent(
            rate_limit_info=RateLimitInfo(status="rejected"),
            uuid="rejected",
            session_id="session-1",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session-1",
        )

    monkeypatch.setattr("claude_agent_sdk.query", query)
    chunks = [chunk async for chunk in _wrapper(tmp_path).reply_stream("hello")]

    errors = [chunk for chunk in chunks if chunk.chunk_type.value == "error"]
    assert [chunk.chunk for chunk in errors] == ["Rate limit exceeded"]


@pytest.mark.asyncio
async def test_reply_stream_uses_error_result_as_terminal_error(tmp_path, monkeypatch):
    """The SDK's non-zero process exit does not duplicate an emitted error result."""
    from claude_agent_sdk import ResultMessage

    async def query(**_kwargs):
        yield ResultMessage(
            subtype="error_during_execution",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="session-1",
            errors=["tool failed"],
        )
        raise RuntimeError("Claude Code returned an error result: tool failed")

    monkeypatch.setattr("claude_agent_sdk.query", query)
    chunks = [chunk async for chunk in _wrapper(tmp_path).reply_stream("hello")]

    errors = [chunk for chunk in chunks if chunk.chunk_type.value == "error"]
    assert [chunk.chunk for chunk in errors] == ["tool failed"]


@pytest.mark.asyncio
async def test_reply_stream_does_not_hide_unrelated_error_after_error_result(tmp_path, monkeypatch):
    """Only the SDK's exact trailing process error is suppressed."""
    from claude_agent_sdk import ResultMessage

    async def query(**_kwargs):
        yield ResultMessage(
            subtype="error_during_execution",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="session-1",
            errors=["tool failed"],
        )
        raise RuntimeError("unrelated store failure")

    monkeypatch.setattr("claude_agent_sdk.query", query)

    with pytest.raises(RuntimeError, match="unrelated store failure"):
        _ = [chunk async for chunk in _wrapper(tmp_path).reply_stream("hello")]


@pytest.mark.asyncio
async def test_reply_stream_reports_session_mirror_errors(tmp_path, monkeypatch):
    """The latest SDK's non-fatal mirror failures remain visible to callers."""
    from claude_agent_sdk import MirrorErrorMessage, ResultMessage

    async def query(**_kwargs):
        yield MirrorErrorMessage(
            subtype="mirror_error",
            data={},
            key={"project_key": "project", "session_id": "session-1"},
            error="disk full",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session-1",
        )

    monkeypatch.setattr("claude_agent_sdk.query", query)
    chunks = [chunk async for chunk in _wrapper(tmp_path).reply_stream("hello")]

    diagnostics = [chunk for chunk in chunks if chunk.metadata.get("event") == "session_mirror_error"]
    assert len(diagnostics) == 1
    assert diagnostics[0].chunk_type == ChunkEnum.DATA
    assert diagnostics[0].chunk == "Session mirror failed: disk full"
    assert not any(chunk.chunk_type == ChunkEnum.ERROR for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper_factory",
    [
        _wrapper,
        lambda tmp_path: AsAgentWrapper(as_llm="", app_context=ApplicationContext(workspace_dir=str(tmp_path))),
    ],
)
@pytest.mark.parametrize("schema", [{}, {"type": "object"}])
async def test_reply_stream_rejects_output_schema(tmp_path, wrapper_factory, schema):
    """Streaming wrappers reject structured-output schemas consistently."""
    wrapper = wrapper_factory(tmp_path)

    with pytest.raises(NotImplementedError, match="Structured output is not supported"):
        await anext(wrapper.reply_stream("hello", output_schema=schema))
