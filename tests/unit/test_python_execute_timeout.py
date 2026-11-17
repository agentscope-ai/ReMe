"""Timeout bounds at the Python executor and HTTP request boundaries."""

from unittest.mock import AsyncMock

import pytest
from agentscope.tool import ExecResult
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reme.components.application_context import ApplicationContext
from reme.components.job.base_job import BaseJob
from reme.components.service.http_service import HttpService
from reme.steps.common.python_execute import PythonExecuteStep


@pytest.fixture(name="backend")
def mock_backend(monkeypatch):
    """Capture deadlines without executing untrusted code or waiting for them."""
    instance = AsyncMock()
    instance.exec_shell.return_value = ExecResult(exit_code=0, stdout=b"ok\n", stderr=b"")
    monkeypatch.setattr("reme.steps.common.python_execute.file_sandbox_for_step", lambda _step: instance)
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["python_timeout", "timeout"])
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (0.25, 0.25),
        ("12.5", 12.5),
        (86400, 86400),
        (86401, 86400),
        ("inf", 86400),
        (float("inf"), 86400),
        ("1e9999", 86400),
        (10**400, 86400),
    ],
)
async def test_timeout_is_capped_before_execution(backend, key, requested, expected):
    """Both supported request keys use a finite deadline and report the effective value."""
    response = await PythonExecuteStep()(code="pass", **{key: requested})
    assert response.success
    assert backend.exec_shell.await_args.kwargs["timeout"] == expected
    assert response.metadata["python_timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("requested", ["nan", float("nan"), "-inf", 0, -1, -(10**400), True, None, "bad"])
async def test_invalid_timeout_never_executes(backend, requested):
    """Invalid inputs fail before constructing or invoking an execution backend."""
    response = await PythonExecuteStep()(code="pass", python_timeout=requested)
    assert not response.success
    assert response.answer == "python_timeout must be a positive number"
    backend.exec_shell.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(("maximum", "expected"), [(30, 30), (120, 60), (100000, 60)])
async def test_default_timeout_respects_server_cap(backend, maximum, expected):
    """An omitted timeout retains 60 seconds unless the server requires less."""
    response = await PythonExecuteStep(max_python_timeout=maximum)(code="pass")
    assert response.success
    assert backend.exec_shell.await_args.kwargs["timeout"] == expected
    assert response.metadata["python_timeout"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("maximum", [0, -1, "nan", "-inf", None, True])
async def test_invalid_server_cap_fails_closed(backend, maximum):
    """A misconfigured server limit cannot silently become unlimited."""
    response = await PythonExecuteStep(max_python_timeout=maximum)(code="pass")
    assert not response.success
    assert response.answer == "max_python_timeout must be a positive number"
    backend.exec_shell.assert_not_awaited()


@pytest.mark.parametrize(("maximum", "expected"), [(86400, 86400), (30, 30)])
@pytest.mark.parametrize("key", ["python_timeout", "timeout"])
def test_http_request_cannot_raise_server_timeout_cap(tmp_path, backend, maximum, expected, key):
    """Exercise the real HTTP job route, including extra fields that bypass job schemas."""
    context = ApplicationContext(workspace_dir=str(tmp_path))
    job = BaseJob(name="python_review", app_context=context)
    job.step_specs = [(PythonExecuteStep, {"app_context": context, "max_python_timeout": maximum})]
    service = HttpService()
    service.service = FastAPI()
    service._add_json_job(job)  # pylint: disable=protected-access
    with TestClient(service.service) as client:
        response = client.post(
            "/python_review",
            json={"code": "pass", key: "inf", "max_python_timeout": "inf"},
        )
    assert response.status_code == 200
    assert response.json()["success"]
    assert response.json()["metadata"]["python_timeout"] == expected
    assert backend.exec_shell.await_args.kwargs["timeout"] == expected
