"""Streaming job for real-time output delivery."""

import asyncio

from .base_job import BaseJob
from ..runtime_context import RuntimeContext
from ...enumeration import ChunkEnum


class StreamJob(BaseJob):
    """Job that streams execution results in real-time.

    Unlike BaseJob which returns a final response, StreamJob pushes
    intermediate results to a queue as they are produced, allowing
    clients to receive updates incrementally.
    """

    async def __call__(self, **kwargs) -> asyncio.Queue:
        """Execute all steps with streaming enabled.

        Args:
            **kwargs: Parameters passed to the runtime context.

        Returns:
            An asyncio.Queue containing streamed chunks.
        """
        context = RuntimeContext(stream=True, **kwargs)
        try:
            for step in self.steps:
                await step(context)
        except Exception as e:
            await context.add_stream_string(str(e), ChunkEnum.ERROR)

        await context.add_stream_done()
        return context.stream_queue
