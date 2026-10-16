"""Direct workspace lifecycle without a Docker daemon."""

# pylint: disable=missing-function-docstring,protected-access,wrong-import-position

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

DockerError = pytest.importorskip("aiodocker.exceptions").DockerError
from deploy.benchmark_docker.direct_docker_workspace import DirectDockerWorkspace  # noqa: E402


def make_workspace():
    workspace = DirectDockerWorkspace(workspace_id="cleanup-test")
    workspace._container = SimpleNamespace(kill=AsyncMock(), wait=AsyncMock(), delete=AsyncMock())
    workspace._client = SimpleNamespace(close=AsyncMock())
    workspace._backend = SimpleNamespace(exec_shell=AsyncMock())
    workspace.is_alive = True
    return workspace


@pytest.mark.asyncio
async def test_stop_kills_and_confirms_exit_before_removal():
    workspace = make_workspace()
    container = workspace._container
    await workspace.stop()
    container.kill.assert_awaited_once_with(signal="SIGKILL")
    container.wait.assert_awaited_once_with(condition="not-running")
    container.delete.assert_not_awaited()
    assert workspace._backend is not None
    assert not workspace.is_alive
    await workspace.stop()
    assert container.kill.await_count == 1
    await workspace.close()
    container.delete.assert_awaited_once_with(force=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["kill", "delete"])
async def test_docker_cleanup_timeout_retains_handles(monkeypatch, operation):
    from deploy.benchmark_docker import direct_docker_workspace

    monkeypatch.setattr(direct_docker_workspace, "DOCKER_CLEANUP_TIMEOUT", 0.01)
    workspace = make_workspace()
    container = workspace._container

    async def blocked(**_kwargs):
        await asyncio.Event().wait()

    getattr(container, operation).side_effect = blocked
    with pytest.raises(TimeoutError):
        await (workspace.stop() if operation == "kill" else workspace.close())
    assert workspace._container is container
    getattr(container, operation).side_effect = None
    await workspace.close()


@pytest.mark.asyncio
async def test_delete_failure_retains_handles_and_can_be_retried():
    workspace = make_workspace()
    container, client, backend = workspace._container, workspace._client, workspace._backend
    container.delete.side_effect = [RuntimeError("delete failed"), None]

    with pytest.raises(RuntimeError, match="delete failed"):
        await workspace.close()
    assert workspace._container is container
    assert workspace._client is client
    assert workspace._backend is backend
    assert not workspace.is_alive
    client.close.assert_not_awaited()
    with pytest.raises(RuntimeError, match="cleanup must complete"):
        await workspace.initialize()

    await workspace.close()
    await workspace.close()
    assert container.delete.await_count == 2
    container.delete.assert_awaited_with(force=True)
    client.close.assert_awaited_once()
    assert workspace._container is workspace._client is workspace._backend is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 500])
async def test_only_missing_container_is_a_successful_delete(status):
    workspace = make_workspace()
    container, client = workspace._container, workspace._client
    container.delete.side_effect = DockerError(status, {"message": "delete failed"})
    if status == 404:
        await workspace.close()
        assert workspace._container is None
        client.close.assert_awaited_once()
    else:
        with pytest.raises(DockerError):
            await workspace.close()
        assert workspace._container is container
        client.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_close_failure_retries_without_deleting_again():
    workspace = make_workspace()
    container, client = workspace._container, workspace._client
    client.close.side_effect = [RuntimeError("client close failed"), None]
    with pytest.raises(RuntimeError, match="client close failed"):
        await workspace.close()
    assert workspace._container is workspace._backend is None
    assert workspace._client is client

    await workspace.close()
    container.delete.assert_awaited_once_with(force=True)
    assert client.close.await_count == 2
    assert workspace._client is None


@pytest.mark.asyncio
async def test_cancelled_delete_retains_state_for_retry():
    workspace = make_workspace()
    container, client = workspace._container, workspace._client
    started = asyncio.Event()

    async def block_delete(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    container.delete.side_effect = block_delete
    task = asyncio.create_task(workspace.close())
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert workspace._container is container
    assert workspace._client is client

    container.delete.side_effect = None
    await workspace.close()
    assert workspace._container is workspace._client is None


@pytest.mark.asyncio
async def test_initialization_failure_keeps_primary_error_and_retries_cleanup(monkeypatch):
    workspace = make_workspace()
    container = workspace._container
    container.delete.side_effect = [RuntimeError("delete failed"), None]
    client, backend = workspace._client, workspace._backend
    workspace._container = workspace._client = workspace._backend = None
    workspace.is_alive = False

    async def provision():
        workspace._container, workspace._client, workspace._backend = container, client, backend

    monkeypatch.setattr(workspace, "_provision_backend", provision)
    backend.exec_shell.return_value = SimpleNamespace(ok=lambda: False, stderr=b"mkdir failed")
    with pytest.raises(RuntimeError, match="mkdir failed") as failure:
        await workspace.initialize()
    assert any("delete failed" in note for note in failure.value.__notes__)
    assert workspace._container is container
    await workspace.close()
    assert workspace._container is workspace._client is workspace._backend is None
