"""Host and candidate version guards match the supported AgentScope runtime."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import pytest

from deploy.benchmark_docker import ImageCandidate
from deploy.benchmark_docker.docker_sandbox import (
    DockerReMeSandbox,
    SandboxCommandError,
    _default_workspace_builder,
)


@pytest.mark.parametrize("version", ["2.0.9", "2.0.10"])
def test_host_version_guard(monkeypatch, version):
    """The configured dependency can create a workspace; stale hosts fail early."""
    pytest.importorskip("aiodocker")
    monkeypatch.setattr("agentscope.__version__", version)
    if version == "2.0.9":
        with pytest.raises(RuntimeError, match="requires agentscope 2.0.10"):
            _default_workspace_builder("version-test", "test-image", {})
    else:
        workspace = _default_workspace_builder("version-test", "test-image", {})
        assert workspace.base_image == "test-image"


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["2.0.9", "2.0.10"])
async def test_candidate_version_guard(monkeypatch, version):
    """Candidate validation rejects stale images even on a supported host."""
    sandbox = DockerReMeSandbox(
        case_id="version-test",
        container_id="version-test",
        candidate=ImageCandidate("test-image"),
        workspace=None,
        config="benchmark.yaml",
        plugins=(),
        command_timeout=30,
        environment_names=[],
    )
    validation = {"agentscope": version, "plugins": {}}
    monkeypatch.setattr(
        sandbox,
        "_exec_checked",
        AsyncMock(return_value=SimpleNamespace(stdout=json.dumps(validation).encode())),
    )
    if version == "2.0.9":
        with pytest.raises(SandboxCommandError, match="requires agentscope 2.0.10"):
            await sandbox._validate_candidate("python")  # pylint: disable=protected-access
    else:
        await sandbox._validate_candidate("python")  # pylint: disable=protected-access
