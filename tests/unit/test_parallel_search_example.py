"""Offline tests for the opt-in Parallel MCP example."""

import argparse
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    client = object()
    transport_calls = []
    client_calls = []

    def transport_factory(**kwargs):
        transport_calls.append(kwargs)
        return transport

    def client_factory(**kwargs):
        client_calls.append(kwargs)
        return client

    monkeypatch.setattr(example, "StreamableHttpTransport", transport_factory)
    monkeypatch.setattr(example, "MCPClient", client_factory)

    result = example.create_client()

    assert transport_calls == [{"url": "https://mcp.parallel.ai/mcp"}]
    assert client_calls == [{"transport": transport, "timeout": 60}]
    assert result is client


def test_validate_url_requires_public_http_url():
    """Only absolute HTTP(S) fetch targets are accepted."""
    assert example.validate_url("https://example.com/article") == "https://example.com/article"
    assert example.validate_url("http://example.com/article") == "http://example.com/article"
    with pytest.raises(argparse.ArgumentTypeError, match="absolute http"):
        example.validate_url("file:///etc/passwd")
    with pytest.raises(argparse.ArgumentTypeError, match="absolute http"):
        example.validate_url("example.com/article")


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


class RateLimitError(Exception):
    """Fake HTTP error carrying the provider's 429 status."""

    def __init__(self):
        super().__init__("too many requests")
        self.response = SimpleNamespace(status_code=429)


def test_main_reports_provider_rate_limit(monkeypatch, capsys):
    """A provider 429 gives the user actionable rate-limit guidance."""
    client = FakeClient(actions=[{"name": "web_search"}, {"name": "web_fetch"}])

    async def fail(*_args, **_kwargs):
        raise RateLimitError()

    monkeypatch.setattr(example, "create_client", lambda: client)
    monkeypatch.setattr(example, "run_example", fail)

    assert example.main(["topic", "https://example.com"]) == 1
    assert "rate limit" in capsys.readouterr().err.lower()
