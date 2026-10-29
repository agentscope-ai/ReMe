"""Observe the first initialized sandbox without counting failed startup attempts."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from deploy.benchmark_docker import DockerReMeSandboxFactory, ImageCandidate
from deploy.benchmark_docker import docker_sandbox


@pytest.mark.asyncio
async def test_start_callback_only_once_after_initialization(monkeypatch):
    """Initialization is the boundary, including replacement workers and partial pool failure."""
    initialized, reported = [], []

    async def initialize(case):
        await asyncio.sleep(0)
        initialized.append(case.case_id)
        if case.case_id == "bad":
            raise RuntimeError("startup failed")

    monkeypatch.setattr(docker_sandbox.DockerReMeSandbox, "initialize", initialize)
    workspace = AsyncMock()
    factory = DockerReMeSandboxFactory(
        ImageCandidate("fixture"),
        workspace_builder=lambda *_: workspace,
        on_sandbox_started=lambda: reported.append(list(initialized)),
    )
    with pytest.raises(RuntimeError, match="startup failed"):
        await factory.create_case("bad")
    assert not reported
    await factory.create_case("good")
    assert reported == [["bad", "good"]]
    await factory.create_case("replacement")
    assert len(reported) == 1


@pytest.mark.asyncio
async def test_start_callback_failure_closes_initialized_resource(monkeypatch):
    """Persisting a start marker must not orphan an already-created workspace on failure."""
    monkeypatch.setattr(docker_sandbox.DockerReMeSandbox, "initialize", AsyncMock())
    workspace = AsyncMock()

    def fail():
        raise OSError("start marker unavailable")

    factory = DockerReMeSandboxFactory(
        ImageCandidate("fixture"),
        workspace_builder=lambda *_: workspace,
        on_sandbox_started=fail,
    )
    with pytest.raises(OSError, match="start marker unavailable"):
        await factory.create_case("created")
    workspace.close.assert_awaited_once()
    assert not factory._sandbox_started_reported  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_partial_pool_start_reports_first_success_before_failure(monkeypatch):
    """An initialized worker counts even if a subsequent worker prevents pool startup."""
    workspaces, started = [], []

    def workspace_builder(*_args):
        workspace = AsyncMock()
        workspaces.append(workspace)
        return workspace

    async def initialize(case):
        if case.case_id == "bad":
            raise ConnectionError("second worker failed")

    monkeypatch.setattr(docker_sandbox.DockerReMeSandbox, "initialize", initialize)
    factory = DockerReMeSandboxFactory(
        ImageCandidate("fixture"),
        workspace_builder=workspace_builder,
        on_sandbox_started=lambda: started.append(True),
    )
    with pytest.raises(ConnectionError, match="second worker failed"):
        await factory.create_cases(["good", "bad"], concurrency=1)
    assert started == [True]
    assert len(workspaces) == 2
    for workspace in workspaces:
        workspace.close.assert_awaited_once()
