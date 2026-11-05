"""Tests for HTTP client display formatting."""

# pylint: disable=protected-access

import json

import httpx
import pytest

from reme.components.client.http_client import HttpClient
from reme.constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT
from reme.enumeration import ChunkEnum


def test_default_client_uses_loopback_address():
    """Clients connect to loopback when no service address is configured."""
    client = HttpClient()

    assert client.base_url == f"http://{REME_DEFAULT_HOST}:{REME_DEFAULT_PORT}"


def test_client_converts_wildcard_bind_address_to_loopback():
    """A service's wildcard bind address is not advertised as a destination."""
    client = HttpClient(host="0.0.0.0", port=8123)

    assert client.base_url == "http://127.0.0.1:8123"


def test_format_for_display_hides_metadata_by_default():
    """Response metadata is available structurally but not shown in normal CLI output."""
    client = HttpClient()
    text = '{"answer":"0.4.0.7","success":true,"metadata":{"version":"0.4.0.7"}}'

    assert client._format_for_display(text) == "0.4.0.7\n✅"


def test_format_for_display_shows_metadata_when_requested():
    """show_metadata=true opts into metadata display."""
    client = HttpClient(show_metadata=True)
    text = '{"answer":"0.4.0.7","success":true,"metadata":{"version":"0.4.0.7"}}'

    assert client._format_for_display(text) == '0.4.0.7\n✅ {"version": "0.4.0.7"}'


@pytest.mark.asyncio
@pytest.mark.parametrize("will_retry", [True, False, None, "false"])
async def test_raw_stream_preserves_errors_and_done_but_cli_raises_terminal_errors(will_retry):
    """A temporary Codex reconnect must not prevent CLI callers from receiving the answer."""
    events = [
        {"chunk_type": "error", "chunk": "upstream error", "metadata": {"will_retry": will_retry}},
        {"chunk_type": "content", "chunk": "recovered answer"},
        {"chunk_type": "done", "done": True},
    ]
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    def handle(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    client = HttpClient()
    client.client = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(handle))
    async with client:
        chunks = [chunk async for chunk in client.stream_chunks("stream_llm_demo")]
        assert [chunk.chunk_type for chunk in chunks] == [ChunkEnum.ERROR, ChunkEnum.CONTENT, ChunkEnum.DONE]
        assert chunks[0].metadata == {"will_retry": will_retry}
        assert chunks[1].chunk == "recovered answer"
        if will_retry is True:
            assert "recovered answer" in "".join([text async for text in client._execute("stream_llm_demo", {})])
        else:
            with pytest.raises(RuntimeError, match="upstream error"):
                _ = [text async for text in client._execute("stream_llm_demo", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed", "interrupted", "waiting"])
async def test_http_collection_and_replay_preserve_complete_outcomes(status):
    """Raw HTTP consumption preserves failures and native metadata all the way through DONE."""
    from reme.schema import StreamChunk, collect_reply, replay_reply

    events = [
        StreamChunk(chunk_type=ChunkEnum.REPLY_START, status="running", chunk=None),
        StreamChunk(chunk="draft", block_id="text"),
        StreamChunk(chunk="corrected", event="end", block_id="text", metadata={"native": {"signature": "signed"}}),
        StreamChunk(
            chunk_type=ChunkEnum.REPLY_END,
            event="end",
            status=status,
            chunk={
                "session_id": "session",
                "result": "corrected",
                "structured_output": [],
                "usage": None,
                "status": status,
                "error": "failure" if status == "failed" else None,
                "raw": {"cache": 7, "future": False},
            },
        ),
        StreamChunk(chunk_type=ChunkEnum.DONE, done=True),
    ]
    if status == "failed":
        events.insert(1, StreamChunk(chunk_type=ChunkEnum.ERROR, chunk="failure"))
    body = "".join(f"data: {event.model_dump_json()}\n\n" for event in events)

    def handle(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    client = HttpClient()
    client.client = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(handle))
    async with client:
        reply = await collect_reply(client.stream_chunks("chat"))
    assert reply["status"] == status
    assert reply["events"] == [event.model_dump(mode="json") for event in events]
    assert await collect_reply(replay_reply(json.loads(json.dumps(reply)))) == reply


@pytest.mark.asyncio
@pytest.mark.parametrize("body", ["data: invalid\n\n", 'data: {"chunk":"partial"}\n\n'])
async def test_raw_http_stream_rejects_corruption_or_missing_done(body):
    """Broken streams must not silently lose events or become successful replies."""

    def handle(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    client = HttpClient()
    client.client = httpx.AsyncClient(base_url=client.base_url, transport=httpx.MockTransport(handle))
    async with client:
        with pytest.raises((ValueError, RuntimeError)):
            _ = [chunk async for chunk in client.stream_chunks("chat")]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "interrupted", "waiting"])
@pytest.mark.parametrize("error", [None, "provider refused the request"])
async def test_cli_reports_incomplete_terminal_outcomes(monkeypatch, status, error):
    """A partial answer followed by DONE must not hide an incomplete invocation."""
    from reme.schema import StreamChunk

    async def stream(_action, _payload):
        yield StreamChunk(chunk="partial answer")
        yield StreamChunk(
            chunk_type=ChunkEnum.REPLY_END,
            event="end",
            status=status,
            chunk={"status": status, "error": error},
        )
        yield StreamChunk(chunk_type=ChunkEnum.DONE, done=True)

    client = HttpClient()
    monkeypatch.setattr(client, "_iter_stream_chunks", stream)
    output = []
    with pytest.raises(RuntimeError, match=error or status):
        async for text in client._execute("chat", {}):
            output.append(text)
    assert output == ["partial answer"]
