"""Tests for ``ChannelNotifyStep`` — workspace-watcher batch → channel event."""

import asyncio
from pathlib import Path

from reme.components.application_context import ApplicationContext
from reme.components.service.mcp_service import ChannelSink
from reme.components.runtime_context import RuntimeContext
from reme.steps.channel.channel_notify import ChannelNotifyStep


class _StubSession:
    """Capture ``send_message`` payloads instead of writing them to a transport."""

    def __init__(self) -> None:
        self.sent: list = []

    async def send_message(self, message) -> None:
        """Record the outbound ``SessionMessage`` for later assertions."""
        self.sent.append(message)


def _run(coro):
    """Drive a coroutine on a fresh event loop (tests don't share one)."""
    return asyncio.new_event_loop().run_until_complete(coro)


def _ctx(changes: list[dict]) -> RuntimeContext:
    """Build a ``RuntimeContext`` pre-populated with the step's ``changes`` input."""
    ctx = RuntimeContext()
    ctx["changes"] = changes
    return ctx


def _app_ctx_with_sink(workspace: Path, stub: _StubSession | None) -> tuple[ApplicationContext, ChannelSink | None]:
    """Build an ``ApplicationContext`` rooted at ``workspace``; attach a sink bound to ``stub`` if given."""
    app_ctx = ApplicationContext(workspace_dir=str(workspace), app_name="reme-test")
    if stub is None:
        return app_ctx, None
    sink = ChannelSink()
    sink.bind(stub)
    app_ctx.metadata["channel_sink"] = sink
    return app_ctx, sink


def test_emits_one_event_per_batch_with_relative_paths(tmp_path):
    """A batch of changes → exactly one notification with relative paths and a count meta."""
    workspace = tmp_path
    (workspace / "resource" / "2026-06-03").mkdir(parents=True)
    f1 = workspace / "resource" / "2026-06-03" / "a.md"
    f1.write_text("x")

    stub = _StubSession()
    app_ctx, _ = _app_ctx_with_sink(workspace, stub)

    step = ChannelNotifyStep(app_context=app_ctx)
    response = _run(
        step(
            context=_ctx(
                [
                    {"change": "added", "path": str(f1)},
                    {"change": "modified", "path": str(f1)},
                ],
            ),
        ),
    )

    assert response.success is True
    assert len(stub.sent) == 1
    params = stub.sent[0].message.root.params
    assert params["meta"] == {"kind": "workspace_change", "count": "2"}
    assert "added: resource/2026-06-03/a.md" in params["content"]
    assert "modified: resource/2026-06-03/a.md" in params["content"]


def test_noop_when_no_changes(tmp_path):
    """An empty changes list must not produce any notification."""
    stub = _StubSession()
    app_ctx, _ = _app_ctx_with_sink(tmp_path, stub)
    step = ChannelNotifyStep(app_context=app_ctx)
    response = _run(step(context=_ctx([])))
    assert response.success is True
    assert not stub.sent


def test_noop_when_sink_not_bound(tmp_path):
    """Step must run cleanly when no ``ChannelSink`` is configured."""
    app_ctx, _ = _app_ctx_with_sink(tmp_path, None)
    step = ChannelNotifyStep(app_context=app_ctx)
    # Should run without raising even though channel_sink is absent from metadata
    response = _run(step(context=_ctx([{"change": "added", "path": "/tmp/x.md"}])))
    assert response.success is True


def test_path_outside_workspace_passes_through_as_is(tmp_path):
    """Paths not under the workspace are emitted verbatim instead of crashing."""
    stub = _StubSession()
    app_ctx, _ = _app_ctx_with_sink(tmp_path, stub)
    step = ChannelNotifyStep(app_context=app_ctx)
    _run(
        step(
            context=_ctx([{"change": "added", "path": "/elsewhere/wild.md"}]),
        ),
    )
    # Stray absolute path → emitted verbatim, no crash, still one event
    assert len(stub.sent) == 1
    assert "added: /elsewhere/wild.md" in stub.sent[0].message.root.params["content"]
