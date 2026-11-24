"""Forward job stream chunks and manage the producer's lifetime."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from ..enumeration import ChunkEnum
from ..schema import StreamChunk
from .logger_utils import get_logger


async def execute_stream_task(
    stream_queue: asyncio.Queue[StreamChunk],
    task: asyncio.Task[Any],
    task_name: str | None = None,
) -> AsyncGenerator[StreamChunk, None]:
    """Yield queued chunks while monitoring the producer and cancel it on exit."""
    logger = get_logger()
    consumer: asyncio.Task[StreamChunk] | None = None
    try:
        while True:
            consumer = get_chunk = asyncio.create_task(stream_queue.get())
            done, _pending = await asyncio.wait({get_chunk, task}, return_when=asyncio.FIRST_COMPLETED)

            # Producer still running — relay the next chunk and continue.
            if task not in done:
                chunk = get_chunk.result()
                yield chunk
                if chunk.done:
                    return
                continue

            # Producer finished. Capture any pending chunk, then stop the consumer wait
            # so we can inspect task state safely.
            pending_chunk: StreamChunk | None = None
            if get_chunk in done:
                pending_chunk = get_chunk.result()
            else:
                get_chunk.cancel()
                try:
                    await get_chunk
                except asyncio.CancelledError:
                    pass

            # Surface task failure first — an exception trumps trailing data.
            if task.cancelled():
                msg = f"Task cancelled: {task_name}" if task_name else "Task cancelled"
                raise asyncio.CancelledError(msg)
            exc = task.exception()
            if exc is not None:
                log_msg = f"Task error in {task_name}: {exc}" if task_name else f"Task error: {exc}"
                logger.error(log_msg, exc_info=exc)
                raise exc

            # Producer ended cleanly — flush pending + drain queue so no chunk is lost,
            # then emit the terminal sentinel.
            if pending_chunk is not None:
                yield pending_chunk
                if pending_chunk.done:
                    return
            while not stream_queue.empty():
                chunk = stream_queue.get_nowait()
                yield chunk
                if chunk.done:
                    return

            yield StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True)
            return

    finally:
        # Cancel consumer wait if still pending (e.g. on consumer aclose).
        if consumer is not None and not consumer.done():
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass
        # Cancel producer task if still running to avoid resource leaks.
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
