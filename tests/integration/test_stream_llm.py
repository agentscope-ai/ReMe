"""Integration tests: stream Agent output through StreamLLMDemoStep.

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real LLM API.
"""

import asyncio
import sys
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _workspace_fixture import workspace_env  # noqa: E402

from reme.enumeration import ChunkEnum  # noqa: E402
from reme.schema import StreamChunk  # noqa: E402
from reme.steps.common.stream_llm_demo import StreamLLMDemoStep  # noqa: E402
from reme.utils.common_utils import execute_stream_task  # noqa: E402

# StreamLLMDemoStep always advertises the ``add`` job as a tool. That job only
# ships in demo.yaml, so inject it into the default config the fixture loads.
ADD_JOB = {
    "backend": "base",
    "description": "add two numbers",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "first addend"},
            "b": {"type": "number", "description": "second addend"},
        },
        "required": ["a", "b"],
    },
    "steps": [{"backend": "add_step"}],
}


async def _test_stream_llm_basic_chat():
    """StreamLLMDemoStep streams text chunks via add_stream_string."""
    with workspace_env() as env:
        app = await env.make_app(jobs={"add": ADD_JOB})
        try:
            step = StreamLLMDemoStep(app_context=app.context)
            queue: asyncio.Queue = asyncio.Queue()
            chunks: list[StreamChunk] = []

            task = asyncio.create_task(
                step(
                    stream_queue=queue,
                    query="Explain step by step how to compute 1 + 1, and give the final answer.",
                ),
            )

            print("\n[stream_basic] streaming output:")
            async for raw in execute_stream_task(queue, task, output_format="chunk"):
                chunk: StreamChunk = raw  # type: ignore[assignment]
                chunks.append(chunk)
                if chunk.chunk_type == ChunkEnum.CONTENT:
                    sys.stdout.write(chunk.chunk)
                    sys.stdout.flush()

            response = task.result()

            # Should have received multiple CONTENT chunks for a longer response
            content_chunks = [c for c in chunks if c.chunk_type == ChunkEnum.CONTENT]
            print(f"\n\n[stream_basic] got {len(content_chunks)} CONTENT chunks")
            assert len(content_chunks) > 1, "Expected multiple CONTENT chunks for streaming"

            # Final answer should be populated
            text = (response.answer or "").strip()
            assert text, "Empty assistant response"
            assert "2" in text, f"Expected '2' in response, got: {text!r}"

            # Concatenated stream text should match the final answer
            streamed_text = "".join(c.chunk for c in content_chunks)
            assert streamed_text.strip() == text, f"Stream text mismatch: {streamed_text!r} vs {text!r}"
            print("✓ test_stream_llm_basic_chat passed")
        finally:
            await env.close_all()


async def _test_stream_llm_with_tool():
    """StreamLLMDemoStep streams tool call events when tools are used."""
    with workspace_env() as env:
        app = await env.make_app(jobs={"add": ADD_JOB})
        try:
            step = StreamLLMDemoStep(app_context=app.context)
            queue: asyncio.Queue = asyncio.Queue()
            chunks: list[StreamChunk] = []

            task = asyncio.create_task(
                step(
                    stream_queue=queue,
                    query="Use the add tool to compute 21 + 21 and report the result.",
                    sys_prompt="Use the `add` tool whenever the user asks to add numbers.",
                ),
            )

            print("\n[stream_tool] streaming output:")
            async for raw in execute_stream_task(queue, task, output_format="chunk"):
                chunk: StreamChunk = raw  # type: ignore[assignment]
                chunks.append(chunk)
                if chunk.chunk_type == ChunkEnum.CONTENT:
                    sys.stdout.write(chunk.chunk)
                    sys.stdout.flush()
                elif chunk.chunk_type == ChunkEnum.TOOL_CALL:
                    sys.stdout.write(f"\033[33m{chunk.chunk}\033[0m")
                    sys.stdout.flush()
                elif chunk.chunk_type == ChunkEnum.TOOL_RESULT:
                    sys.stdout.write(f"\033[32m{chunk.chunk}\033[0m")
                    sys.stdout.flush()

            response = task.result()

            tool_call_chunks = [c for c in chunks if c.chunk_type == ChunkEnum.TOOL_CALL]
            tool_result_chunks = [c for c in chunks if c.chunk_type == ChunkEnum.TOOL_RESULT]
            content_chunks = [c for c in chunks if c.chunk_type == ChunkEnum.CONTENT]

            print(f"\n\n[stream_tool] TOOL_CALL chunks: {len(tool_call_chunks)}")
            print(f"[stream_tool] TOOL_RESULT chunks: {len(tool_result_chunks)}")
            print(f"[stream_tool] CONTENT chunks: {len(content_chunks)}")

            assert len(tool_call_chunks) > 0, "Expected TOOL_CALL chunks"
            assert len(tool_result_chunks) > 0, "Expected TOOL_RESULT chunks"

            text = (response.answer or "").strip()
            print(f"[stream_tool] final answer: {text!r}")
            assert "42" in text, f"Expected '42' in response, got: {text!r}"
            print("✓ test_stream_llm_with_tool passed")
        finally:
            await env.close_all()


async def _test_stream_llm_fallback_no_stream():
    """Without stream_queue, still uses streaming under the hood for real-time output."""
    with workspace_env() as env:
        app = await env.make_app(jobs={"add": ADD_JOB})
        try:
            step = StreamLLMDemoStep(app_context=app.context)
            queue: asyncio.Queue = asyncio.Queue()
            chunks: list[StreamChunk] = []

            task = asyncio.create_task(
                step(
                    stream_queue=queue,
                    query="Explain step by step how to compute 1 + 1, and give the final answer.",
                ),
            )

            print("\n[fallback_stream] streaming output:")
            async for raw in execute_stream_task(queue, task, output_format="chunk"):
                chunk: StreamChunk = raw  # type: ignore[assignment]
                chunks.append(chunk)
                if chunk.chunk_type == ChunkEnum.CONTENT:
                    sys.stdout.write(chunk.chunk)
                    sys.stdout.flush()
                elif chunk.chunk_type == ChunkEnum.THINK:
                    sys.stdout.write(f"\033[2m{chunk.chunk}\033[0m")
                    sys.stdout.flush()

            response = task.result()
            text = (response.answer or "").strip()
            content_chunks = [c for c in chunks if c.chunk_type == ChunkEnum.CONTENT]
            print(f"\n\n[fallback_stream] got {len(content_chunks)} CONTENT chunks")
            assert text, "Empty assistant response"
            assert "2" in text, f"Expected '2' in response, got: {text!r}"
            print("✓ test_stream_llm_fallback_no_stream passed")
        finally:
            await env.close_all()


def test_stream_llm_basic_chat():
    """StreamLLMDemoStep streams text chunks via add_stream_string."""
    asyncio.run(_test_stream_llm_basic_chat())


def test_stream_llm_with_tool():
    """StreamLLMDemoStep streams tool call events when tools are used."""
    asyncio.run(_test_stream_llm_with_tool())


def test_stream_llm_fallback_no_stream():
    """Without stream_queue, falls back to non-streaming reply."""
    asyncio.run(_test_stream_llm_fallback_no_stream())


async def _demo_stream_print():
    """Real-time streaming print demo — ask a longer question to see chunked output."""
    with workspace_env() as env:
        app = await env.make_app(jobs={"add": ADD_JOB})
        try:
            step = StreamLLMDemoStep(app_context=app.context)
            queue: asyncio.Queue = asyncio.Queue()

            query = (
                "Please explain in detail how neural networks learn through backpropagation. "
                "Include the chain rule, gradient descent, and give a concrete example with numbers."
            )

            task = asyncio.create_task(
                step(
                    stream_queue=queue,
                    query=query,
                    sys_prompt="You are a knowledgeable AI teacher. Explain concepts thoroughly.",
                ),
            )

            async for raw in execute_stream_task(queue, task, output_format="chunk"):
                chunk: StreamChunk = raw  # type: ignore[assignment]
                if chunk.chunk_type == ChunkEnum.CONTENT:
                    sys.stdout.write(chunk.chunk)
                    sys.stdout.flush()
                elif chunk.chunk_type == ChunkEnum.THINK:
                    sys.stdout.write(f"\033[2m{chunk.chunk}\033[0m")
                    sys.stdout.flush()
                elif chunk.chunk_type == ChunkEnum.TOOL_CALL:
                    sys.stdout.write(f"\n\033[33m[tool_call] {chunk.chunk}\033[0m")
                    sys.stdout.flush()
                elif chunk.chunk_type == ChunkEnum.TOOL_RESULT:
                    sys.stdout.write(f"\033[32m{chunk.chunk}\033[0m")
                    sys.stdout.flush()
            print()
        finally:
            await env.close_all()


async def _run_all():
    print("=== StreamLLMDemoStep integration tests ===")
    await _test_stream_llm_basic_chat()
    await _test_stream_llm_with_tool()
    await _test_stream_llm_fallback_no_stream()
    print("\nAll stream integration tests passed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(_demo_stream_print())
    else:
        asyncio.run(_run_all())
