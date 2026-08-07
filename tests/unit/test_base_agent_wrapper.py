"""Tests for shared agent wrapper behavior."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reme.components.agent_wrapper import (
    AsAgentWrapper,
    BaseAgentWrapper,
    CcAgentWrapper,
    CodexAgentWrapper,
    handle_session_command,
)
from reme.components.agent_wrapper.as_agent_wrapper import WorkspaceBackend
from reme.components.agent_wrapper import as_agent_wrapper
from reme.components.agent_wrapper import base_agent_wrapper
from reme.components.application_context import ApplicationContext
from reme.components.outbound_proxy import FixedHttpOutboundProxy
from reme.components import base_component
from reme.enumeration import ComponentEnum


class _VersionedAgentWrapper(BaseAgentWrapper):
    SDK_PACKAGE = "example-agent-sdk"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compacted_session = ""

    async def reply(self, inputs, **kwargs) -> dict:
        return {"inputs": inputs, "kwargs": kwargs}

    async def compact_session(self, session_id: str) -> None:
        self.compacted_session = session_id


def test_init_logs_sdk_version(monkeypatch):
    """An SDK-backed wrapper logs its installed distribution version."""
    logger = MagicMock()
    logger.bind.return_value = logger
    monkeypatch.setattr(base_component, "get_logger", lambda: logger)
    monkeypatch.setattr(base_agent_wrapper.metadata, "version", lambda package: "1.2.3")

    _VersionedAgentWrapper(name="versioned")

    logger.info.assert_called_once_with("Agent SDK name=versioned package=example-agent-sdk version=1.2.3")


def test_init_logs_unknown_when_sdk_distribution_metadata_is_missing(monkeypatch):
    """Missing distribution metadata does not prevent wrapper initialization."""
    logger = MagicMock()
    logger.bind.return_value = logger
    monkeypatch.setattr(base_component, "get_logger", lambda: logger)

    def missing_version(package):
        raise base_agent_wrapper.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(base_agent_wrapper.metadata, "version", missing_version)

    _VersionedAgentWrapper()

    logger.info.assert_called_once_with(
        "Agent SDK name=_VersionedAgentWrapper package=example-agent-sdk version=unknown",
    )


@pytest.mark.asyncio
async def test_session_commands_are_backend_neutral():
    """Session commands work independently from a chat transport."""
    wrapper = _VersionedAgentWrapper()

    assert await handle_session_command(wrapper, "hello", "session-1") is None
    assert (await handle_session_command(wrapper, "/clear", "session-1")).session_id is None
    unavailable = await handle_session_command(wrapper, "/compact", None)
    assert unavailable.answer == "No active conversation to compact."

    compacted = await handle_session_command(wrapper, "/compact", "session-1")
    assert compacted.session_id == "session-1"
    assert wrapper.compacted_session == "session-1"


@pytest.mark.parametrize(
    ("wrapper_class", "sdk_package"),
    [
        (AsAgentWrapper, "agentscope"),
        (CcAgentWrapper, "claude-agent-sdk"),
        (CodexAgentWrapper, "openai-codex"),
    ],
)
def test_agent_wrappers_declare_sdk_package(wrapper_class, sdk_package):
    """Each concrete backend identifies the distribution that provides its SDK."""
    assert wrapper_class.SDK_PACKAGE == sdk_package


def test_project_path_is_independent_from_runtime_workspace(tmp_path):
    """Project assets can live outside the runtime workspace."""
    workspace = tmp_path / "project" / ".reme"
    wrapper = _VersionedAgentWrapper(
        app_context=ApplicationContext(workspace_dir=str(workspace)),
        project_path="..",
    )

    assert wrapper.workspace_path == workspace
    assert wrapper.project_path == tmp_path / "project"
    assert wrapper.cwd == tmp_path / "project"
    assert wrapper.project_skills_root == tmp_path / "project" / "skills"


@pytest.mark.parametrize("wrapper_class", [AsAgentWrapper, CcAgentWrapper, CodexAgentWrapper])
def test_agent_wrappers_share_project_skill_resolution(tmp_path, wrapper_class):
    """Every backend resolves selected skills through the base project root."""
    workspace = tmp_path / "project" / ".reme"
    skill = tmp_path / "project" / "skills" / "one"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# one", encoding="utf-8")
    kwargs = {
        "app_context": ApplicationContext(workspace_dir=str(workspace)),
        "project_path": "..",
    }
    if wrapper_class is AsAgentWrapper:
        kwargs["as_llm"] = ""
    wrapper = wrapper_class(**kwargs)

    assert wrapper._resolve_project_skills(["one", "one"]) == {"one": skill}  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_agentscope_backend_passes_configured_environment_to_bash(tmp_path, monkeypatch):
    """AgentScope subprocesses receive config environment values explicitly."""
    monkeypatch.setenv("REME_AGENT_ENV_TEST", "parent")
    backend = WorkspaceBackend(str(tmp_path), {"REME_AGENT_ENV_TEST": "configured"})

    result = await backend.exec_shell(
        [sys.executable, "-c", "import os; print(os.environ['REME_AGENT_ENV_TEST'])"],
        cwd=str(tmp_path),
    )

    assert result.exit_code == 0
    assert result.stdout == b"configured\n"


@pytest.mark.asyncio
async def test_agentscope_bash_uses_managed_proxy_without_changing_subprocess_environment(tmp_path):
    """AgentScope applies the managed proxy only to its command backend."""
    context = ApplicationContext(
        workspace_dir=str(tmp_path),
        environment={"TOOL_ENV": "preserved"},
    )
    proxy = FixedHttpOutboundProxy(url="http://127.0.0.1:18080")
    await proxy.start()
    context.components = {ComponentEnum.OUTBOUND_PROXY: {"default": proxy}}
    wrapper = AsAgentWrapper(app_context=context, as_llm="")

    await wrapper.start()
    bash = wrapper._builtin_tools(["bash"])[0]  # pylint: disable=protected-access
    backend = bash._backend  # pylint: disable=protected-access

    assert wrapper.subprocess_environment == {"TOOL_ENV": "preserved"}
    assert "HTTP_PROXY" not in wrapper.subprocess_environment
    assert backend._environment["TOOL_ENV"] == "preserved"  # pylint: disable=protected-access
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        assert backend._environment[key] == proxy.http_url  # pylint: disable=protected-access

    await wrapper.close()
    await proxy.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply_kwargs", "expected"),
    [
        ({}, []),
        ({"use_builtin_tools": True}, "all"),
        ({"builtin_tools": ["read"]}, ["read"]),
    ],
)
async def test_agentscope_builtin_tools_are_opt_in(tmp_path, monkeypatch, reply_kwargs, expected):
    """AgentScope loads no built-in tools unless a caller explicitly opts in."""
    wrapper = AsAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)), as_llm="")
    wrapper.as_llm = SimpleNamespace(model=object())
    observed = {}

    def builtin_tools(names, *, sequential_tool_calls=False):
        observed["names"] = names
        observed["sequential_tool_calls"] = sequential_tool_calls
        return []

    class FakeAgent:
        """Minimal constructor double for AgentScope Agent."""

        def __init__(self, **kwargs):
            self.state = kwargs["state"]

    monkeypatch.setattr(wrapper, "_builtin_tools", builtin_tools)
    monkeypatch.setattr(as_agent_wrapper, "Agent", FakeAgent)

    await wrapper._build_agent("hello", **reply_kwargs)  # pylint: disable=protected-access

    assert observed == {"names": expected, "sequential_tool_calls": True}
