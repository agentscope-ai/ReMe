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
async def test_stream_retry_notifications_continue_but_terminal_errors_raise(will_retry):
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
        if will_retry is True:
            chunks = [chunk async for chunk in client.stream_chunks("stream_llm_demo")]
            assert [chunk.chunk_type for chunk in chunks] == [ChunkEnum.ERROR, ChunkEnum.CONTENT]
            assert chunks[0].metadata == {"will_retry": True}
            assert chunks[-1].chunk == "recovered answer"
        else:
            with pytest.raises(RuntimeError, match="upstream error"):
                _ = [chunk async for chunk in client.stream_chunks("stream_llm_demo")]
