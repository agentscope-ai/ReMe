"""Streaming job for real-time output delivery."""

from .base_job import BaseJob
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ChunkEnum


@R.register("stream")
class StreamJob(BaseJob):
    """Job that streams results to a queue in real-time."""

    async def __call__(self, **kwargs):
        context = RuntimeContext(stream=True, **kwargs)
        try:
            for step in self.step_components:
                await step(context)
        except Exception as e:
            await context.add_stream_string(str(e), ChunkEnum.ERROR)
        await context.add_stream_done()
