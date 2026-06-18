"""Streaming job for real-time output delivery."""

from .base_job import BaseJob
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ChunkEnum


@R.register("stream")
class StreamJob(BaseJob):
    """Job that streams chunks to a queue instead of returning a Response."""

    async def __call__(self, **kwargs) -> None:
        """Run steps; emit failures as ERROR chunks, then a terminal DONE marker."""
        merged = {**self.kwargs, **kwargs}
        context = RuntimeContext(**merged)
        try:
            for step in self._build_steps():
                await step(context)
        except Exception as e:
            await context.add_stream_string(str(e), ChunkEnum.ERROR)
        # Always emit DONE so consumers can detach even after an error.
        await context.add_stream_done()
