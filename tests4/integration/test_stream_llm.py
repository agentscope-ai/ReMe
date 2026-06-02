"""Integration tests: stream Agent output through StreamLLMDemoStep.

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real LLM API.
"""

import asyncio
import os
import tempfile

from reme4 import Application
from reme4.config import resolve_app_config
from reme4.enumeration import ChunkEnum
from reme4.schema import StreamChunk
from reme4.steps.common.stream_llm_demo import StreamLLMDemoStep
from reme4.utils import load_env

load_env()


class _temp_chdir:
    """chdir to path for the duration of the block; restore on exit."""

    def __init__(self, path):
        self.path = path
        self._old = None

    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self._old)


async def _make_app() -> Application:
    """Build and start an Application from the default config."""
    cfg = resolve_app_config(log_to_console=False, log_to_file=False, enable_logo=False)
    app = Application(**cfg)
    await app.start()
    return app


async def _test_stream_llm_basic_chat():
    """StreamLLMDemoStep streams text chunks via add_stream_string."""
    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        app = await _make_app()
        try:
            step = StreamLLMDemoStep(app_context=app.context)
            queue: asyncio.Queue = asyncio.Queue()

            response = await step(
                stream_queue=queue,
                query="What is 1 + 1? Reply with just the number.",
            )

            # Collect all chunks from the queue
            chunks = []
            while not queue.empty():
                chunks.append(await queue.get())

            # Should have received CONTENT chunks
            content_chunks = [c for c in chunks if c.chunk_type == ChunkEnum.CONTENT]
            print(f"\n[stream_basic] got {len(content_chunks)} CONTENT chunks")
            assert len(content_chunks) > 0, "Expected at least one CONTENT chunk"

            # Final answer should be populated
            text = (response.answer or "").strip()
            print(f"[stream_basic] final answer: {text!r}")
            assert text, "Empty assistant response"
            assert "2" in text, f"Expected '2' in response, got: {text!r}"

            # Concatenated stream text should match the final answer
            streamed_text = "".join(c.chunk for c in content_chunks)
            assert streamed_text.strip() == text, f"Stream text mismatch: {streamed_text!r} vs {text!r}"
            print("✓ test_stream_llm_basic_chat passed")
        finally:
            await app.close()


async def _test_stream_llm_with_tool():
    """StreamLLMDemoStep streams tool call events when tools are used."""
    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        app = await _make_app()
        try:
            step = StreamLLMDemoStep(app_context=app.context)
            queue: asyncio.Queue = asyncio.Queue()

            response = await step(
                stream_queue=queue,
                query="Use the add tool to compute 21 + 21 and report the result.",
                sys_prompt="Use the `add` tool whenever the user asks to add numbers.",
                use_add_tool=True,
            )

            # Collect all chunks
            chunks = []
            while not queue.empty():
                chunks.append(await queue.get())

            tool_call_chunks = [c for c in chunks if c.chunk_type == ChunkEnum.TOOL_CALL]
            tool_result_chunks = [c for c in chunks if c.chunk_type == ChunkEnum.TOOL_RESULT]
            content_chunks = [c for c in chunks if c.chunk_type == ChunkEnum.CONTENT]

            print(f"\n[stream_tool] TOOL_CALL chunks: {len(tool_call_chunks)}")
            print(f"[stream_tool] TOOL_RESULT chunks: {len(tool_result_chunks)}")
            print(f"[stream_tool] CONTENT chunks: {len(content_chunks)}")

            assert len(tool_call_chunks) > 0, "Expected TOOL_CALL chunks"
            assert len(tool_result_chunks) > 0, "Expected TOOL_RESULT chunks"

            text = (response.answer or "").strip()
            print(f"[stream_tool] final answer: {text!r}")
            assert "42" in text, f"Expected '42' in response, got: {text!r}"
            print("✓ test_stream_llm_with_tool passed")
        finally:
            await app.close()


async def _test_stream_llm_fallback_no_stream():
    """Without stream_queue, falls back to non-streaming reply."""
    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        app = await _make_app()
        try:
            step = StreamLLMDemoStep(app_context=app.context)
            response = await step(
                query="What is 1 + 1? Reply with just the number.",
            )
            text = (response.answer or "").strip()
            print(f"\n[fallback] response: {text!r}")
            assert text, "Empty assistant response"
            assert "2" in text, f"Expected '2' in response, got: {text!r}"
            print("✓ test_stream_llm_fallback_no_stream passed")
        finally:
            await app.close()


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
    import sys  # pylint: disable=import-outside-toplevel,redefined-outer-name

    with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
        app = await _make_app()
        try:
            step = StreamLLMDemoStep(app_context=app.context)
            queue: asyncio.Queue = asyncio.Queue()

            query = (
                "Please explain in detail how neural networks learn through backpropagation. "
                "Include the chain rule, gradient descent, and give a concrete example with numbers."
            )

            async def consumer():
                """Print chunks to terminal in real-time."""
                while True:
                    chunk = await queue.get()
                    if chunk.done:
                        break
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

            consumer_task = asyncio.create_task(consumer())

            await step(
                stream_queue=queue,
                query=query,
                sys_prompt="You are a knowledgeable AI teacher. Explain concepts thoroughly.",
            )
            # Signal done so consumer exits
            await queue.put(StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True))
            await consumer_task
        finally:
            await app.close()


async def _run_all():
    print("=== StreamLLMDemoStep integration tests ===")
    await _test_stream_llm_basic_chat()
    await _test_stream_llm_with_tool()
    await _test_stream_llm_fallback_no_stream()
    print("\nAll stream integration tests passed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(_demo_stream_print())
    else:
        asyncio.run(_run_all())
