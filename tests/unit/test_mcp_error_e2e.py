"""E2E test: verify MCPService signals ReMe job failures as MCP errors.

Tests the real ``MCPService.add_job`` → ``execute_tool`` → ``Tool.run()``
path.  When a BaseJob returns ``Response(success=False, ...)`` the
``execute_tool`` closure must raise ``ToolError`` so FastMCP produces an
MCP response with ``isError: true``.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError

from reme.components.job import BaseJob
from reme.components.service import MCPService
from reme.schema import Response

# ---------------------------------------------------------------------------
# Real ReMe jobs that mirror the auto_memory_codex error pattern
# ---------------------------------------------------------------------------


class _SucceedJob(BaseJob):
    """A job that returns success=True (the normal path)."""

    def _build_steps(self):
        return []

    async def __call__(self, **kwargs) -> Response:
        return Response(success=True, answer="Recorded 3 facts into daily/2026-07-25/notes.md")


class _FailJob(BaseJob):
    """A job that returns success=False — exactly what auto_memory_codex does
    when it cannot resolve the transcript path."""

    def _build_steps(self):
        return []

    async def __call__(self, **kwargs) -> Response:
        return Response(success=False, answer="Error: could not resolve transcript path")


class _RealisticFailJob(BaseJob):
    """A job that mimics auto_memory_codex more closely: sets response in the
    same way the real step does."""

    def _build_steps(self):
        return []

    async def __call__(self, **kwargs) -> Response:
        # Same pattern as auto_memory_codex.py:64-66
        resp = Response()
        resp.success = False
        resp.answer = "Error: could not resolve transcript path"
        return resp


# ---------------------------------------------------------------------------
# Minimal app object needed by MCPService.build_service
# ---------------------------------------------------------------------------


def _dummy_app(name: str = "test") -> SimpleNamespace:
    """Minimal object needed by MCPService.build_service."""
    return SimpleNamespace(
        config=SimpleNamespace(app_name=name),
        context=SimpleNamespace(metadata={}),
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestMCPServiceToolError:
    """Tests that exercise the real MCPService.add_job + execute_tool path."""

    @pytest.mark.asyncio
    async def test_success_job_returns_answer(self):
        """A successful ReMe job → execute_tool returns the answer text."""
        service = MCPService(tool_error_on_failure=True)
        service.build_service(_dummy_app("success-test"))

        job = _SucceedJob(name="test_succeed")
        assert service.add_job(job) is True

        # Call the tool through FastMCP's tool registry — same as a real MCP call.
        tool = await service.service.get_tool("test_succeed")
        result = await tool.run({})
        assert result.is_error is False, f"Success job should have is_error=False, got {result}"
        content_text = result.content[0].text
        assert "Recorded 3 facts" in content_text, f"Wrong content: {content_text}"

    @pytest.mark.asyncio
    async def test_fail_job_raises_tool_error(self):
        """A failed ReMe job (success=False) → execute_tool raises ToolError.

        This is THE test for the P2 bug.  Before the fix, MCPService returned
        the error text as a normal result and _result_status logged it as 'ok'.
        """
        service = MCPService(tool_error_on_failure=True)
        service.build_service(_dummy_app("fail-test"))

        job = _FailJob(name="test_fail")
        assert service.add_job(job) is True

        tool = await service.service.get_tool("test_fail")

        # run() should raise ToolError (which FastMCP converts to isError=True)
        with pytest.raises(ToolError) as exc_info:
            await tool.run({})
        assert "could not resolve transcript path" in str(exc_info.value), f"Wrong error message: {exc_info.value}"

    @pytest.mark.asyncio
    async def test_realistic_fail_job_raises_tool_error(self):
        """Even when the step sets response.success/answer manually (the
        auto_memory_codex pattern), execute_tool must raise ToolError."""
        service = MCPService(tool_error_on_failure=True)
        service.build_service(_dummy_app("realistic-test"))

        job = _RealisticFailJob(name="test_realistic_fail")
        assert service.add_job(job) is True

        tool = await service.service.get_tool("test_realistic_fail")

        with pytest.raises(ToolError) as exc_info:
            await tool.run({})
        assert "could not resolve transcript path" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_both_jobs_in_same_service(self):
        """Success and failure coexist, no cross-talk."""
        service = MCPService(tool_error_on_failure=True)
        service.build_service(_dummy_app("both-test"))

        assert service.add_job(_SucceedJob(name="test_ok")) is True
        assert service.add_job(_FailJob(name="test_err")) is True

        ok_tool = await service.service.get_tool("test_ok")
        err_tool = await service.service.get_tool("test_err")

        # Success
        ok_result = await ok_tool.run({})
        assert ok_result.is_error is False

        # Failure
        with pytest.raises(ToolError) as exc_info:
            await err_tool.run({})
        assert "could not resolve transcript path" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Hook classification — verifies _result_status handles the real MCP shape
# ---------------------------------------------------------------------------


def test_hook_result_status_on_real_response_shape():
    """The hook's _result_status correctly classifies real MCP response dicts."""
    # pylint: disable=import-outside-toplevel
    _HOOK_DIR = Path(__file__).parents[2] / "plugins" / "codex" / "reme" / "hooks"
    import sys

    sys.path.insert(0, str(_HOOK_DIR))
    import auto_memory  # noqa: E402  # pylint: disable=wrong-import-position

    # Real shape from a failed job: ToolError → CallToolResult(isError=True)
    failed = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [{"type": "text", "text": "Error: could not resolve transcript path"}],
            "isError": True,
        },
    }
    assert auto_memory._result_status(failed) == "error"

    ok = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [{"type": "text", "text": "Recorded 3 facts."}],
            "isError": False,
        },
    }
    assert auto_memory._result_status(ok) == "ok"

    skipped = {
        "jsonrpc": "2.0",
        "id": 2,
        "result": {
            "content": [{"type": "text", "text": "Skipped: no messages"}],
            "isError": False,
        },
    }
    assert auto_memory._result_status(skipped) == "skipped"
