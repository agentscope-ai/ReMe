"""Common utility functions for the application.

Provides general-purpose utilities including text hashing and async
stream processing for task execution.
"""

import asyncio
import hashlib
from typing import AsyncGenerator, Literal

from .logger_utils import get_logger
from ..enumeration import ChunkEnum
from ..schema import StreamChunk

logger = get_logger()


def hash_text(text: str, encoding: str = "utf-8") -> str:
    """Generate SHA-256 hash of text content.

    Creates a cryptographic hash suitable for content identification
    and deduplication purposes.

    Args:
        text: Input text to hash.
        encoding: Character encoding for the text. Defaults to "utf-8".

    Returns:
        Hexadecimal string representation of the SHA-256 hash
        (64 characters).
    """
    return hashlib.sha256(text.encode(encoding)).hexdigest()


async def execute_stream_task(
        stream_queue: asyncio.Queue,
        task: asyncio.Task,
        task_name: str | None = None,
        output_format: Literal["str", "bytes", "chunk"] = "str",
) -> AsyncGenerator[str | bytes | StreamChunk, None]:
    """Core stream flow execution logic.

    Handles streaming from a queue while monitoring the task completion.
    Properly manages errors and resource cleanup.

    This async generator yields streaming data from a background task,
    handling the coordination between queue-based communication and
    task lifecycle management. It ensures proper cleanup even when
    exceptions occur.

    Args:
        stream_queue: Queue to receive StreamChunk objects from the
            background task.
        task: Background asyncio Task executing the flow. This task
            will be monitored for completion and exceptions.
        task_name: Optional flow name for logging purposes. Used in
            error messages for debugging.
        output_format: Output format control:
            - "str": SSE-formatted string (e.g., "data:{json}\\n\\n")
            - "bytes": SSE-formatted bytes for HTTP responses
            - "chunk": Raw StreamChunk objects for further processing

    Yields:
        - str: SSE-formatted data when output_format="str"
        - bytes: SSE-formatted data when output_format="bytes"
        - StreamChunk: Raw chunk objects when output_format="chunk"

    Raises:
        Exception: Re-raises any exception from the background task.
    """
    try:
        while True:
            # Wait for next chunk or check if task failed
            get_chunk = asyncio.create_task(stream_queue.get())
            done, _pending = await asyncio.wait(
                {get_chunk, task}, return_when=asyncio.FIRST_COMPLETED
            )

            # Priority 1: Check if main task finished (may have exception)
            if task in done:
                # Task finished - check for exceptions first
                exc = task.exception()
                if exc:
                    log_msg = f"Task error in {task_name}: {exc}" if task_name else f"Task error: {exc}"
                    logger.exception(log_msg)
                    raise exc

                # Task completed successfully - drain remaining chunks if any
                if get_chunk in done:
                    chunk: StreamChunk = get_chunk.result()
                    if output_format == "chunk":
                        yield chunk
                        if chunk.done:
                            break
                    else:
                        if chunk.done:
                            yield b"data:[DONE]\n\n" if output_format == "bytes" else "data:[DONE]\n\n"
                            break
                        data = f"data:{chunk.model_dump_json()}\n\n"
                        yield data.encode() if output_format == "bytes" else data
                else:
                    # No more chunks, task completed
                    get_chunk.cancel()
                    if output_format == "chunk":
                        yield StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True)
                    else:
                        yield b"data:[DONE]\n\n" if output_format == "bytes" else "data:[DONE]\n\n"
                    break

            elif get_chunk in done:
                # Got a chunk from the queue (task still running)
                chunk: StreamChunk = get_chunk.result()

                # Handle raw chunk mode
                if output_format == "chunk":
                    yield chunk
                    if chunk.done:
                        break
                    continue

                # Handle SSE format mode (str or bytes)
                if chunk.done:
                    yield b"data:[DONE]\n\n" if output_format == "bytes" else "data:[DONE]\n\n"
                    break

                data = f"data:{chunk.model_dump_json()}\n\n"
                yield data.encode() if output_format == "bytes" else data

    finally:
        # Ensure task is canceled if still running to avoid resource leaks
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
