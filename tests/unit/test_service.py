"""Tests for service job registration behavior."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reme.components.job import BaseJob, StreamJob
from reme.components.service import MCPService
from reme.schema import Response


def _dummy_app():
    """Minimal object needed by MCPService.build_service."""

    async def start():
        return None

    async def close():
        return None

    return SimpleNamespace(
        config=SimpleNamespace(app_name="test"),
        context=SimpleNamespace(metadata={}),
        start=start,
        close=close,
    )


def _app_with_jobs(**jobs):
    """Minimal object needed by BaseService.add_jobs."""
    return SimpleNamespace(context=SimpleNamespace(jobs=jobs))


def test_service_registers_all_enabled_jobs_by_default():
    """Omitting service.jobs preserves registration of every service-enabled job."""
    service = MCPService()
    service.add_job = Mock(return_value=True)
    enabled = BaseJob(name="enabled")
    disabled = BaseJob(name="disabled", enable_serve=False)

    service.add_jobs(_app_with_jobs(enabled=enabled, disabled=disabled))

    service.add_job.assert_called_once_with(enabled)


def test_service_jobs_restricts_registration_to_configured_names():
    """service.jobs acts as a whitelist without overriding enable_serve."""
    service = MCPService(jobs=["selected"])
    service.add_job = Mock(return_value=True)
    selected = BaseJob(name="selected")
    unselected = BaseJob(name="unselected")
    disabled = BaseJob(name="disabled", enable_serve=False)

    service.add_jobs(
        _app_with_jobs(selected=selected, unselected=unselected, disabled=disabled),
    )

    service.add_job.assert_called_once_with(selected)


def test_empty_service_jobs_disables_job_registration():
    """An explicit empty whitelist exposes no jobs."""
    service = MCPService(jobs=[])
    service.add_job = Mock(return_value=True)

    service.add_jobs(_app_with_jobs(enabled=BaseJob(name="enabled")))

    service.add_job.assert_not_called()


def test_explicit_service_jobs_reject_missing_disabled_and_unsupported_jobs():
    """An explicit service.jobs list fails instead of starting an incomplete service."""
    missing_service = MCPService(jobs=["missing"])
    with pytest.raises(KeyError, match="missing"):
        missing_service.add_jobs(_app_with_jobs())

    disabled_service = MCPService(jobs=["disabled"])
    with pytest.raises(ValueError, match="disabled"):
        disabled_service.add_jobs(_app_with_jobs(disabled=BaseJob(name="disabled", enable_serve=False)))

    stream_service = MCPService(jobs=["stream"])
    stream_service.add_job = Mock(return_value=False)
    with pytest.raises(TypeError, match="stream"):
        stream_service.add_jobs(_app_with_jobs(stream=StreamJob(name="stream")))


def test_mcp_service_registers_job_with_empty_parameters():
    """Empty job parameters must remain a dict for FastMCP FunctionTool validation."""
    service = MCPService()
    service.build_service(_dummy_app())

    job = BaseJob(name="empty_params", parameters={})

    assert service.add_job(job) is True


def test_mcp_service_reports_stream_job_skipped():
    """MCPService intentionally does not expose StreamJob tools."""
    service = MCPService()
    service.build_service(_dummy_app())

    job = StreamJob(name="stream")

    assert service.add_job(job) is False


class _RecordingJob:
    """Small callable matching the job contract used by MCPService.add_job."""

    name = "record"
    description = "Record arguments"
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}}

    def __init__(self, response: Response | None = None):
        self.response = response or Response(answer="ok")
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_mcp_service_injects_job_kwargs_and_rejects_conflicts():
    """Configured job arguments are injected exactly once and remain server-owned."""

    async def run():
        service = MCPService(injected_job_kwargs={"tool_context_id": "ctx-1"})
        service.build_service(_dummy_app())
        job = _RecordingJob()
        job.parameters = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "tool_context_id": {"type": "string"},
            },
            "required": ["query", "tool_context_id"],
        }
        assert service.add_job(job) is True
        tool = await service.service.get_tool(job.name)
        assert tool is not None
        assert "tool_context_id" not in tool.parameters["properties"]
        assert tool.parameters["required"] == ["query"]

        result = await tool.run({"query": "alpha"})
        assert job.calls == [{"query": "alpha", "tool_context_id": "ctx-1"}]
        assert "ok" in str(result.content)

        with pytest.raises(Exception, match="tool_context_id injected by the MCP server"):
            await tool.run({"query": "alpha", "tool_context_id": "caller"})

    asyncio.run(run())


def test_mcp_service_can_raise_tool_error_for_unsuccessful_response():
    """Configured MCP services translate failed Responses into tool errors."""

    async def run():
        service = MCPService(tool_error_on_failure=True)
        service.build_service(_dummy_app())
        job = _RecordingJob(Response(answer="failed", success=False))
        assert service.add_job(job) is True
        tool = await service.service.get_tool(job.name)
        assert tool is not None

        with pytest.raises(Exception, match="failed"):
            await tool.run({})

    asyncio.run(run())


def test_service_lifespan_closes_app_after_error():
    """Application resources close even when serving exits with an exception."""

    async def run():
        events = []

        async def start():
            events.append("start")

        async def close():
            events.append("close")

        app = SimpleNamespace(start=start, close=close)
        lifespan = MCPService()._lifespan(app, "127.0.0.1", 0)  # pylint: disable=protected-access
        with pytest.raises(RuntimeError, match="stop"):
            async with lifespan(None):
                raise RuntimeError("stop")
        assert events == ["start", "close"]

    asyncio.run(run())
