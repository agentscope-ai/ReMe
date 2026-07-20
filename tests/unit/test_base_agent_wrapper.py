"""Tests for shared agent wrapper behavior."""

from unittest.mock import MagicMock

import pytest

from reme.components.agent_wrapper import AsAgentWrapper, BaseAgentWrapper, CcAgentWrapper, CodexAgentWrapper
from reme.components.agent_wrapper import base_agent_wrapper
from reme.components import base_component


class _VersionedAgentWrapper(BaseAgentWrapper):
    SDK_PACKAGE = "example-agent-sdk"

    async def reply(self, inputs, **kwargs) -> dict:
        return {"inputs": inputs, "kwargs": kwargs}


def test_init_logs_sdk_version(monkeypatch):
    """An SDK-backed wrapper logs its installed distribution version."""
    logger = MagicMock()
    logger.bind.return_value = logger
    monkeypatch.setattr(base_component, "get_logger", lambda: logger)
    monkeypatch.setattr(base_agent_wrapper.metadata, "version", lambda package: "1.2.3")

    _VersionedAgentWrapper(name="versioned")

    logger.info.assert_called_once_with("Agent SDK package=example-agent-sdk version=1.2.3")


def test_init_logs_unknown_when_sdk_distribution_metadata_is_missing(monkeypatch):
    """Missing distribution metadata does not prevent wrapper initialization."""
    logger = MagicMock()
    logger.bind.return_value = logger
    monkeypatch.setattr(base_component, "get_logger", lambda: logger)

    def missing_version(package):
        raise base_agent_wrapper.metadata.PackageNotFoundError(package)

    monkeypatch.setattr(base_agent_wrapper.metadata, "version", missing_version)

    _VersionedAgentWrapper()

    logger.info.assert_called_once_with("Agent SDK package=example-agent-sdk version=unknown")


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
