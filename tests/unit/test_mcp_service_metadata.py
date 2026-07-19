"""Tests for MCP service metadata preservation."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from reme.components.job import BaseJob
from reme.components.service.mcp_service import MCPService
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


def _make_service_and_capture(job):
    """Build an MCPService, register *job*, and return the captured ``execute_tool`` closure.

    ``FastMCP`` has no public ``_tools`` dict.  We patch ``FastMCP.add_tool``
    so that the ``FunctionTool`` passed in is captured rather than stored in
    internal state.  The captured tool's ``.fn`` attribute *is* the
    ``execute_tool`` closure written in ``MCPService.add_job``.
    """
    service = MCPService()
    service.build_service(_dummy_app())

    captured = {}

    original_add_tool = service.service.add_tool

    def _capture_add_tool(tool_obj, **kwargs):
        captured["tool"] = tool_obj
        return original_add_tool(tool_obj, **kwargs)

    with patch.object(service.service, "add_tool", side_effect=_capture_add_tool):
        service.add_job(job)

    return captured["tool"].fn


@pytest.mark.asyncio
async def test_execute_tool_returns_plain_answer_when_metadata_empty():
    """When response.metadata is empty, return the answer string directly."""
    job = BaseJob(name="test_job", parameters={})

    expected = Response(answer="simple answer", success=True, metadata={})

    with patch.object(BaseJob, "__call__", new_callable=AsyncMock, return_value=expected):
        execute_tool = _make_service_and_capture(job)
        result = await execute_tool()

    assert result == "simple answer"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_execute_tool_returns_json_envelope_when_metadata_present():
    """When response.metadata has data, return JSON envelope with answer + metadata."""
    job = BaseJob(name="list_job", parameters={})

    expected = Response(
        answer="Listed 4 file(s) under user-001/",
        success=True,
        metadata={"items": ["a.md", "b.md", "c.md", "d.md"], "count": 4},
    )

    with patch.object(BaseJob, "__call__", new_callable=AsyncMock, return_value=expected):
        execute_tool = _make_service_and_capture(job)
        result = await execute_tool()

    # Result should be a JSON string
    parsed = json.loads(result)
    assert parsed["answer"] == "Listed 4 file(s) under user-001/"
    assert parsed["metadata"]["items"] == ["a.md", "b.md", "c.md", "d.md"]
    assert parsed["metadata"]["count"] == 4


@pytest.mark.asyncio
async def test_execute_tool_handles_non_serializable_metadata():
    """Metadata with non-serializable types should fall back to str()."""
    import datetime

    job = BaseJob(name="edge_job", parameters={})

    expected = Response(
        answer="result",
        success=True,
        metadata={"date": datetime.date(2026, 7, 19)},
    )

    with patch.object(BaseJob, "__call__", new_callable=AsyncMock, return_value=expected):
        execute_tool = _make_service_and_capture(job)
        result = await execute_tool()

    parsed = json.loads(result)
    assert parsed["answer"] == "result"
    assert "2026-07-19" in parsed["metadata"]["date"]
