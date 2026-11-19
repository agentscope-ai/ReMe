"""Exercise the Claude Code relay against a local Anthropic HTTP fixture."""

import asyncio
import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from reme.components.agent_wrapper.cc_agent_wrapper import CcAgentWrapper
from reme.components.anthropic_relay import BailianAnthropicRelay
from reme.components.anthropic_relay.bailian_anthropic_relay import _search_request
from reme.components.application_context import ApplicationContext
from reme.components.service.loopback_http import LoopbackHTTP
from reme.enumeration import ComponentEnum

# pylint: disable=protected-access

SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
BILLING = "x-anthropic-billing-header: cc_version=2.1.325; cc_entrypoint=sdk-py;"
PROMPT = {"type": "text", "text": "Preserve this prompt.", "cache_control": {"type": "ephemeral"}}


@pytest.mark.parametrize("system", [None, "Plain prompt", [PROMPT]])
def test_search_adds_missing_billing_block(system):
    """Add a separate marker block while preserving the existing prompt."""
    body = json.dumps({"system": system, "tools": [SEARCH_TOOL]}).encode()
    output = json.loads(_search_request(body))["system"]
    assert output[0] == {"type": "text", "text": "x-anthropic-billing-header: cc_entrypoint=cli;"}
    expected = [{"type": "text", "text": system}] if isinstance(system, str) else system or []
    assert output[1:] == expected


@pytest.mark.asyncio
async def test_claude_relay_streams_search_and_preserves_other_requests(tmp_path, monkeypatch):
    """Bind the real wrapper and prove SSE streaming, HTTP status and original body forwarding."""
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "existing.example")
    first = b'event: content_block_start\ndata: {"content_block":{"type":"web_search_tool_result"}}\n\n'
    last = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
    release = asyncio.Event()
    requests = []

    async def upstream(request):
        requests.append((request.url, request.headers, await request.body()))
        if request.url.path.endswith("/count_tokens"):
            return Response("retry later", status_code=429, headers={"Retry-After": "2"})

        async def stream():
            yield first
            await asyncio.wait_for(release.wait(), 5)
            yield last

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"request-id": "example"})

    server = LoopbackHTTP(Starlette(routes=[Route("/apps/anthropic/{path:path}", upstream, methods=["POST"])]))
    try:
        context = ApplicationContext(workspace_dir=str(tmp_path))
        relay = BailianAnthropicRelay(base_url=f"{await server.listen()}/apps/anthropic", app_context=context)
        context.components[ComponentEnum.ANTHROPIC_RELAY] = {"default": relay}
        wrapper = CcAgentWrapper(anthropic_relay="default", api_key="fixture", app_context=context)
        async with relay, wrapper, httpx.AsyncClient(trust_env=False) as client:
            opts = wrapper._build_options("search", **wrapper._merged_kwargs({}))
            assert opts.env["ANTHROPIC_BASE_URL"] == relay.endpoint
            assert opts.env["ANTHROPIC_AUTH_TOKEN"] == "fixture"
            assert opts.env["NO_PROXY"] == "existing.example,127.0.0.1"
            assert opts.env["NO_PROXY"] == opts.env["no_proxy"]
            direct = wrapper._build_options("search", base_url="https://other.example")
            assert direct.env["ANTHROPIC_BASE_URL"] == "https://other.example"

            body = {"system": [{"type": "text", "text": BILLING}, PROMPT], "tools": [SEARCH_TOOL], "stream": True}
            async with client.stream(
                "POST",
                f"{relay.endpoint}/v1/messages?beta=true",
                json=body,
                headers={"Authorization": "Bearer fixture", "anthropic-version": "2023-06-01"},
            ) as reply:
                assert reply.status_code == 200
                assert reply.headers["request-id"] == "example"
                assert reply.headers["content-type"] == "text/event-stream; charset=utf-8"
                chunks = reply.aiter_raw()
                assert await asyncio.wait_for(anext(chunks), 5) == first
                release.set()
                assert b"".join([chunk async for chunk in chunks]) == last

            url, headers, forwarded = requests[0]
            assert url.path == "/apps/anthropic/v1/messages" and url.query == "beta=true"
            assert headers["authorization"] == "Bearer fixture"
            assert headers["anthropic-version"] == "2023-06-01"
            assert json.loads(forwarded) == {
                **body,
                "system": [{"type": "text", "text": BILLING.replace("sdk-py", "cli")}, PROMPT],
            }

            raw = b'{ "system": "unmodified", "tools": [] }'
            ordinary = await client.post(f"{relay.endpoint}/v1/messages", content=raw)
            assert ordinary.content == first + last
            assert requests[-1][2] == raw
            unavailable = await client.post(f"{relay.endpoint}/v1/messages/count_tokens", json=body)
            assert unavailable.status_code == 429 and unavailable.headers["retry-after"] == "2"
            assert json.loads(requests[-1][2]) == body

            origin = httpx.URL(relay.endpoint).copy_with(path="/v1/messages")
            assert (await client.post(origin, json=body)).status_code == 404
            assert len(requests) == 3

        with pytest.raises(RuntimeError, match="not running"):
            _ = relay.endpoint
    finally:
        await server.close()
