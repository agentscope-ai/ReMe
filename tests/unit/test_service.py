"""Tests for service job registration behavior."""

from types import SimpleNamespace
from unittest.mock import Mock

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
    service = MCPService(jobs=["selected", "disabled"])
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
