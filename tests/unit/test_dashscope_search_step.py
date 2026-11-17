"""DashScope web search contracts, with no provider requests."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from dashscope import AioMultiModalConversation
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse, MultiModalConversationResponse

from reme.application import Application
from reme.components.job import BaseJob
from reme.config.config_parser import _load_config
from reme.steps.common.dashscope_search import DashScopeSearchStep


@pytest.fixture(name="provider")
def mock_provider(monkeypatch):
    """Supply SDK frames and record stream cleanup."""
    frames, closed = [], []

    async def stream():
        try:
            for frame in frames:
                yield frame
        finally:
            closed.append(True)

    call = AsyncMock(side_effect=lambda **_kwargs: stream())
    monkeypatch.setattr(AioMultiModalConversation, "call", call)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    return frames, closed, call


def chunk(output=None, **kwargs):
    """Build an SDK response, including source-only frames."""
    return MultiModalConversationResponse.from_api_response(
        DashScopeAPIResponse(
            status_code=200,
            request_id="request-1",
            output=output or {},
            **kwargs,
        ),
    )


@pytest.mark.asyncio
async def test_default_job_starts_without_key_and_rejects_call(monkeypatch, tmp_path):
    """Default config loads and the registered job starts without credentials."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    call = AsyncMock(side_effect=AssertionError("must not contact provider"))
    monkeypatch.setattr(AioMultiModalConversation, "call", call)
    config = _load_config("default.yaml")["jobs"]["dashscope_search"]
    app = Application(
        workspace_dir=str(tmp_path),
        service={"backend": "cli"},
        jobs={"dashscope_search": config},
        enable_logo=False,
        log_to_file=False,
    )
    await app.start()
    try:
        response = await app.run_job("dashscope_search", query="latest Qwen models")
        assert not response.success
        assert response.answer == "DASHSCOPE_API_KEY is required"
        assert config["parameters"]["required"] == ["query"]
        call.assert_not_awaited()
    finally:
        await app.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [None, "", "  "])
async def test_empty_query_does_not_call_provider(provider, query):
    """Reject missing or blank queries before contacting DashScope."""
    response = await DashScopeSearchStep()(query=query)
    assert not response.success
    assert response.answer == "query is required"
    provider[2].assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_preserves_sources_and_separates_generated_content(provider):
    """Source-only, repeated, empty, and later source frames retain original fields and order."""
    frames, closed, call = provider
    first = {"index": 1, "title": "One", "url": "https://example.com/1", "unknown": {"raw": True}}
    second = {"index": 2, "title": "Two", "url": "https://example.com/2", "snippet": "provider snippet"}
    frames.extend(
        [
            chunk({"search_info": {"search_results": [first], "provider": "web"}}),
            chunk(
                {
                    "search_info": {"search_results": [dict(first, index=9), second]},
                    "choices": [{"message": {"role": "assistant", "content": [{"text": "Generated "}]}}],
                },
            ),
            chunk(
                {
                    "search_info": {"search_results": []},
                    "choices": [{"message": {"role": "assistant", "content": "content"}}],
                },
                usage={"input_tokens": 12, "output_tokens": 2, "plugins": {"search": {"count": 1}}},
            ),
            chunk(),
        ],
    )
    response = await DashScopeSearchStep(api_key="configured-key")(query=0)

    assert response.success
    assert response.answer == {
        "search_info": {"search_results": [first, second], "provider": "web"},
        "content": "Generated content",
    }
    assert response.metadata == {
        "model": "qwen3.8-flash",
        "request_id": "request-1",
        "usage": {"input_tokens": 12, "output_tokens": 2, "characters": 0, "plugins": {"search": {"count": 1}}},
    }
    assert closed == [True]
    session = call.await_args.kwargs["session"]
    assert session.closed
    call.assert_awaited_once_with(
        api_key="configured-key",
        session=session,
        model="qwen3.8-flash",
        messages=[{"role": "user", "content": [{"text": "0"}]}],
        enable_search=True,
        enable_thinking=False,
        stream=True,
        incremental_output=True,
        search_options={"forced_search": True, "enable_source": True, "search_strategy": "turbo"},
    )


@pytest.mark.asyncio
async def test_empty_results_remain_empty(provider):
    """An ordinary response with no search hits does not invent sources."""
    provider[0].append(chunk({"search_info": {"search_results": []}}))
    response = await DashScopeSearchStep()(query="no hits")
    assert response.success
    assert response.answer == {"search_info": {"search_results": []}, "content": ""}


@pytest.mark.asyncio
async def test_provider_error_after_sources_fails_job_and_closes_stream(provider):
    """A failed stream cannot be published as a successful partial search."""
    frames, closed, _ = provider
    frames.extend(
        [
            chunk({"search_info": {"search_results": [{"url": "https://example.com"}]}}),
            MultiModalConversationResponse(
                status_code=429,
                code="Throttling",
                message="Try later",
                request_id="failed-request",
            ),
        ],
    )
    job = BaseJob()
    job.step_specs = [(DashScopeSearchStep, {})]
    response = await job(query="query")
    assert not response.success
    assert "Throttling: Try later" in response.answer
    assert "failed-request" in response.answer
    assert closed == [True]


@pytest.mark.asyncio
async def test_timeout_closes_stream(monkeypatch):
    """Bound the entire stream, including waiting for another frame."""
    closed = []

    async def stream():
        try:
            yield chunk()
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    monkeypatch.setattr(AioMultiModalConversation, "call", AsyncMock(return_value=stream()))
    with pytest.raises(TimeoutError):
        await DashScopeSearchStep(api_key="test-key", timeout=0.01)(query="query")
    assert closed == [True]
