"""Streaming job for real-time output delivery."""

from .base_job import BaseJob
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ChunkEnum


@R.register("stream")
class StreamJob(BaseJob):
    """Job that streams chunks to a queue instead of returning a Response."""

    async def __call__(self, **kwargs) -> None:
        """Execute steps and stream output; errors are sent as ERROR chunks."""
        context = RuntimeContext(**kwargs)
        try:
            for step in self.step_components:
                await step(context)
        except Exception as e:
            await context.add_stream_string(str(e), ChunkEnum.ERROR)
        await context.add_stream_done()
