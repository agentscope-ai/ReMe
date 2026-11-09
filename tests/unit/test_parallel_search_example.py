"""Offline tests for the opt-in Parallel MCP example."""

import argparse
import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from integrations.parallel_search import example


class FakeClient:
    """Mimic the small public MCPClient surface used by the example."""

    def __init__(self, actions=None, responses=None):
        self.list_actions = AsyncMock(return_value=actions or [])
        self.calls = []
        self.responses = responses or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        return None

    def __call__(self, action, **kwargs):
        self.calls.append((action, kwargs))

        async def stream():
            for response in self.responses.get(action, []):
                yield response

        return stream()


def test_create_client_uses_current_parallel_endpoint(monkeypatch):
    """The custom transport targets Parallel's current public MCP endpoint."""
    transport = object()
    transport_calls = []

    def transport_factory(**kwargs):
        transport_calls.append(kwargs)
        return transport

    monkeypatch.setattr(example, "StreamableHttpTransport", transport_factory)

    result = example.create_client()

    assert transport_calls[0]["url"] == "https://mcp.parallel.ai/mcp"
    assert callable(transport_calls[0]["httpx_client_factory"])
    assert result.transport is transport
    assert result.timeout == 60


def test_validate_url_requires_public_http_url():
    """Only absolute HTTP(S) fetch targets without embedded credentials are accepted."""
    assert example.validate_url("https://example.com/article") == "https://example.com/article"
    assert example.validate_url("http://example.com/article") == "http://example.com/article"
    with pytest.raises(argparse.ArgumentTypeError, match="absolute http"):
        example.validate_url("file:///etc/passwd")
    with pytest.raises(argparse.ArgumentTypeError, match="absolute http"):
        example.validate_url("example.com/article")
    with pytest.raises(argparse.ArgumentTypeError, match="embedded credentials"):
        example.validate_url("https://user:password@example.com/article")


def test_run_example_discovers_and_calls_both_tools_in_order():
    """The example uses the documented tool arguments and prints both results."""
    client = FakeClient(
        actions=[{"name": "web_search"}, {"name": "web_fetch"}],
        responses={"web_search": ["search result"], "web_fetch": ["page content"]},
    )
    output = []

    topic = "latest memory system updates"
    asyncio.run(example.run_example(topic, "https://example.com", client, output.append))

    assert client.list_actions.await_count == 1
    assert client.calls == [
        (
            "web_search",
            {"objective": topic, "search_queries": [topic]},
        ),
        (
            "web_fetch",
            {"urls": ["https://example.com"], "objective": topic},
        ),
    ]
    assert output == [
        f"## Search: {topic}",
        "search result",
        "## Fetch: https://example.com",
        "page content",
    ]


def test_run_example_fails_before_calling_when_tools_are_missing():
    """An incompatible endpoint fails before any tool is dispatched."""
    client = FakeClient(actions=[{"name": "web_search"}])

    with pytest.raises(RuntimeError, match="web_fetch"):
        asyncio.run(example.run_example("topic", "https://example.com", client))

    assert not client.calls


def test_main_reports_rate_limit_from_real_mcp_transport(monkeypatch, capsys):
    """A mocked HTTP 429 survives MCP transport error normalization."""

    def respond(_request):
        return httpx.Response(429, json={"error": "rate limit exceeded"})

    client = example.create_client(httpx_transport=httpx.MockTransport(respond))
    monkeypatch.setattr(example, "create_client", lambda: client)

    assert example.main(["topic", "https://example.com"]) == 1
    assert "rate limit" in capsys.readouterr().err.lower()
    assert client.rate_limited is True
