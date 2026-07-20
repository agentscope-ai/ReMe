"""Tests for the Claude Code agent wrapper."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper
from reme.components.application_context import ApplicationContext
from reme.config import resolve_app_config
from reme.enumeration import ComponentEnum

# pylint: disable=protected-access


def _wrapper(tmp_path: Path) -> CcAgentWrapper:
    return CcAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))


def _skill_roots(tmp_path: Path) -> tuple[Path, Path]:
    return (
        tmp_path / ".claude" / "skills",
        tmp_path / "mem_session" / "claude_config" / "skills",
    )


def test_ensure_claude_skill_dir_adds_selected_skills_without_replacing_existing(tmp_path):
    """Selected workspace skills are added while unrelated Claude skills remain."""
    project_skills = tmp_path / "skills"
    (project_skills / "one").mkdir(parents=True)
    (project_skills / "two").mkdir()
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
    config_dir = tmp_path / "mem_session" / "claude_config"

    _wrapper(tmp_path)._ensure_claude_skill_dir(config_dir, "all")

    for root in _skill_roots(tmp_path):
        assert root.is_dir()
        assert not root.is_symlink()
        assert {path.name for path in root.iterdir()} == {"one", "two"}


def test_ensure_claude_skill_dir_migrates_old_directory_link(tmp_path):
    """A legacy link to the whole project skills directory is migrated safely."""
    project_skills = tmp_path / "skills"
    (project_skills / "one").mkdir(parents=True)
    config_dir = tmp_path / "mem_session" / "claude_config"
    legacy_root = tmp_path / ".claude" / "skills"
    legacy_root.parent.mkdir(parents=True)
    legacy_root.symlink_to(project_skills, target_is_directory=True)

    _wrapper(tmp_path)._ensure_claude_skill_dir(config_dir, ["one"])

    assert legacy_root.is_dir()
    assert not legacy_root.is_symlink()
    assert (legacy_root / "one").resolve() == (project_skills / "one").resolve()


def test_ensure_claude_skill_dir_rejects_paths_as_skill_names(tmp_path):
    """Skill selectors cannot escape the project skills directory."""
    (tmp_path / "skills").mkdir()

    with pytest.raises(ValueError, match="Invalid skill name"):
        _wrapper(tmp_path)._ensure_claude_skill_dir(tmp_path / "config", ["../outside"])


def test_configured_skills_are_added_without_filtering_existing_skills(tmp_path):
    """A selected ReMe skill does not become Claude's runtime skill allowlist."""
    project_skills = tmp_path / "skills"
    (project_skills / "one").mkdir(parents=True)
    (project_skills / "two").mkdir()

    opts = _wrapper(tmp_path)._build_options("hello", skills=["one"])

    assert opts.skills == "all"
    for root in _skill_roots(tmp_path):
        assert (root / "one").resolve() == (project_skills / "one").resolve()
        assert not (root / "two").exists()


def test_system_prompt_mode_replace_preserves_current_behavior(tmp_path):
    """Replace mode passes a string system prompt directly to the SDK."""
    opts = _wrapper(tmp_path)._build_options(
        "hello",
        system_prompt="custom prompt",
        system_prompt_mode="replace",
    )

    assert opts.system_prompt == "custom prompt"


def test_system_prompt_mode_append_uses_claude_code_preset(tmp_path):
    """Append mode retains Claude Code's preset and appends the custom prompt."""
    opts = _wrapper(tmp_path)._build_options(
        "hello",
        system_prompt="custom prompt",
        system_prompt_mode="append",
    )

    assert opts.system_prompt == {
        "type": "preset",
        "preset": "claude_code",
        "append": "custom prompt",
    }


def test_system_prompt_mode_rejects_unknown_value(tmp_path):
    """Invalid prompt modes fail with a clear configuration error."""
    with pytest.raises(ValueError, match="Unknown system_prompt_mode"):
        _wrapper(tmp_path)._build_options("hello", system_prompt_mode="merge")


def test_default_claude_code_system_prompt_mode_is_replace():
    """The built-in configuration preserves the existing replacement behavior."""
    config = resolve_app_config(log_config=False)

    assert config["components"]["agent_wrapper"]["claude_code"]["system_prompt_mode"] == "replace"


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
            credential={"api_key": "default-key", "base_url": "https://default.example.test"},
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
