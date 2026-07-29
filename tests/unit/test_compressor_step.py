"""Unit tests for CompressorStep without LLM dependencies."""

import asyncio

from agentscope.message import TextBlock, ThinkingBlock
from agentscope.model import ChatResponse

from reme.steps.evolve.compressor import CompressorStep


def test_response_text_handles_non_streaming_chat_response():
    """A plain ChatResponse must not crash the async-iteration check.

    Regression: ``ChatResponse`` inherits agentscope's ``DictMixin`` whose
    instance ``__getattr__`` raises ``KeyError``, so ``hasattr(result,
    "__aiter__")`` raised instead of returning False and every compression
    failed with ``KeyError: '__aiter__'``.
    """

    async def run():
        result = ChatResponse(
            content=[
                {"type": "thinking", "thinking": "internal reasoning"},
                {"type": "text", "text": "compressed "},
                {"type": "text", "text": "output"},
            ],
            is_last=True,
        )

        text = await CompressorStep._response_text(result)  # pylint: disable=protected-access

        assert text == "compressed output"

    asyncio.run(run())


def test_response_text_handles_pydantic_content_blocks():
    """Real LLM responses carry pydantic block models, not dicts.

    Regression: ``block.get(...)`` crashed with ``'ThinkingBlock' object has
    no attribute 'get'`` on responses from reasoning models.
    """

    async def run():
        result = ChatResponse(
            content=[
                ThinkingBlock(type="thinking", thinking="internal reasoning"),
                TextBlock(type="text", text="compressed output"),
            ],
            is_last=True,
        )

        text = await CompressorStep._response_text(result)  # pylint: disable=protected-access

        assert text == "compressed output"

    asyncio.run(run())


def test_response_text_consumes_streaming_chunks_and_keeps_last():
    """An async generator of ChatResponses is drained and the last chunk wins."""

    async def run():
        async def stream():
            yield ChatResponse(content=[{"type": "text", "text": "partial"}], is_last=False)
            yield ChatResponse(content=[{"type": "text", "text": "final text"}], is_last=True)

        text = await CompressorStep._response_text(stream())  # pylint: disable=protected-access

        assert text == "final text"

    asyncio.run(run())
