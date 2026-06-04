"""Tests for ``ChannelSink`` — outbound ``notifications/claude/channel`` plumbing.

Strategy: stub a session-like object with an async ``send_message`` capture
list and exercise three paths:

* not bound → emit is a no-op (no exception, no captured message)
* bound + valid meta → captured JSON-RPC notification carries our method
  + content + the meta we passed
* bound + meta with non-identifier keys → the bad keys are dropped, the
  rest passes through verbatim
"""

import asyncio

from reme4.components.service.mcp_service import ChannelSink


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


def test_emit_without_bind_is_noop():
    """Emitting before any session is bound must silently no-op."""
    sink = ChannelSink()
    _run(sink.emit("hello", {"k": "v"}))  # must not raise


def test_emit_after_bind_sends_channel_notification():
    """A bound session receives a JSON-RPC notification carrying content + meta verbatim."""
    sink = ChannelSink()
    stub = _StubSession()
    sink.bind(stub)

    _run(sink.emit("ingest done", {"path": "resource/2026-06-03/x.md", "kind": "ingest"}))

    assert len(stub.sent) == 1
    payload = stub.sent[0].message.root
    assert payload.method == "notifications/claude/channel"
    assert payload.params["content"] == "ingest done"
    assert payload.params["meta"] == {"path": "resource/2026-06-03/x.md", "kind": "ingest"}


def test_emit_filters_non_identifier_meta_keys():
    """Meta keys that aren't pure ``[A-Za-z0-9_]`` identifiers are dropped before send."""
    sink = ChannelSink()
    stub = _StubSession()
    sink.bind(stub)

    _run(
        sink.emit(
            "x",
            {
                "good_key": "ok",
                "bad-key": "dropped",  # hyphen
                "also.bad": "dropped",  # dot
                "Number9": "kept",
            },
        ),
    )

    meta = stub.sent[0].message.root.params["meta"]
    assert meta == {"good_key": "ok", "Number9": "kept"}


def test_unbind_returns_to_noop():
    """After ``unbind``, subsequent emits stop reaching the previously bound session."""
    sink = ChannelSink()
    stub = _StubSession()
    sink.bind(stub)
    sink.unbind()

    _run(sink.emit("x", {}))
    assert not stub.sent


def test_emit_swallows_send_failures():
    """A failing send_message must not bubble out (notification is best-effort)."""

    class _BoomSession:
        """Session whose ``send_message`` always raises, to exercise the failure path."""

        async def send_message(self, message):
            """Raise to simulate a broken transport."""
            raise RuntimeError("transport broke")

    sink = ChannelSink()
    sink.bind(_BoomSession())
    _run(sink.emit("x", {}))  # must not raise
