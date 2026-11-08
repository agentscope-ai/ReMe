"""Streaming job for real-time output delivery."""

from .base_job import BaseJob, _describe_exception
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ChunkEnum


@R.register("stream")
class StreamJob(BaseJob):
    """Job that streams chunks to a queue instead of returning a Response."""

    def __init__(self, *args, stream_queue_size: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(stream_queue_size, bool) or not isinstance(stream_queue_size, int) or stream_queue_size < 0:
            raise ValueError("stream_queue_size must be a nonnegative integer")
        self.stream_queue_size = stream_queue_size

    async def __call__(self, **kwargs) -> None:
        """Run steps; emit failures as ERROR chunks, then a terminal DONE marker."""
        merged = {**self.kwargs, **kwargs}
        context = RuntimeContext(**merged)
        with self._execution(context):
            try:
                await self._run_steps(context)
            except Exception as e:
                context.response.success = False
                context.response.answer = _describe_exception(e)
                await context.add_stream_string(str(e), ChunkEnum.ERROR)
                context.stream_error = context.response.answer
            # Always emit DONE so consumers can detach even after an error.
            await context.add_stream_done()
