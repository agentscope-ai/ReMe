"""Tests for service job registration behavior."""

from types import SimpleNamespace

from reme.components.job import BaseJob, StreamJob
from reme.components.service import MCPService


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
