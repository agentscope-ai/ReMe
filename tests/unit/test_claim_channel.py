"""Tests for ``ClaimChannelStep`` — current MCP session binding."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace

from fastmcp.server.context import _current_context

from reme.components.application_context import ApplicationContext
from reme.components.service.mcp_service import ChannelSink
from reme.steps.channel.claim_channel import ClaimChannelStep


class _StubSession:
    """Capture outbound channel messages after being bound."""

    def __init__(self) -> None:
        self.sent: list = []

    async def send_message(self, message) -> None:
        """Record a message sent by the channel sink."""
        self.sent.append(message)


def _run(coro):
    """Drive a coroutine on a fresh event loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


def test_claim_channel_binds_current_session(tmp_path):
    """The active FastMCP session becomes the sink recipient."""
    app_ctx = ApplicationContext(workspace_dir=str(tmp_path), app_name="reme-test")
    sink = ChannelSink()
    app_ctx.metadata["channel_sink"] = sink
    session = _StubSession()
    ctx = SimpleNamespace(session=session, session_id="sid-1")
    token = _current_context.set(ctx)
    try:
        step = ClaimChannelStep(app_context=app_ctx)
        response = _run(step())
    finally:
        _current_context.reset(token)

    assert response.success is True
    assert response.answer["claimed"] is True
    assert response.answer["session_id"] == "sid-1"
    assert response.metadata["claimed"] is True

    _run(sink.emit("hello", {"kind": "test"}))
    assert len(session.sent) == 1


def test_claim_channel_reports_missing_fastmcp_context(tmp_path):
    """Calling outside a FastMCP request reports a clean unclaimed result."""
    app_ctx = ApplicationContext(workspace_dir=str(tmp_path), app_name="reme-test")
    app_ctx.metadata["channel_sink"] = ChannelSink()
    step = ClaimChannelStep(app_context=app_ctx)

    response = _run(step())

    assert response.success is True
    assert response.answer["claimed"] is False
    assert response.metadata["claimed"] is False
    assert "No active context" in response.answer["reason"]


def test_claim_channel_missing_sink_is_still_controlled_under_optimized_python(tmp_path):
    """Runtime validation must not rely on assert, which Python -O removes."""
    code = f"""
import asyncio
from types import SimpleNamespace
from fastmcp.server.context import _current_context
from reme.components.application_context import ApplicationContext
from reme.steps.channel.claim_channel import ClaimChannelStep

class Session:
    async def send_message(self, message):
        pass

async def main():
    app_ctx = ApplicationContext(workspace_dir={str(tmp_path)!r}, app_name="reme-test")
    ctx = SimpleNamespace(session=Session(), session_id="sid-optimized")
    token = _current_context.set(ctx)
    try:
        response = await ClaimChannelStep(app_context=app_ctx)()
        print(response.answer)
    finally:
        _current_context.reset(token)

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "'claimed': False" in result.stdout
    assert "channel_sink not configured" in result.stdout
