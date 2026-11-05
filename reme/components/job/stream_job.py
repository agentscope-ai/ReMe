"""Streaming job for real-time output delivery."""

from .base_job import BaseJob, _describe_exception
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ChunkEnum


@R.register("stream")
class StreamJob(BaseJob):
    """Job that streams chunks to a queue instead of returning a Response."""

    async def __call__(self, **kwargs) -> None:
        """Run steps; emit failures as ERROR chunks, then a terminal DONE marker."""
        merged = {**self.kwargs, **kwargs}
        workspace = self._bind_workspace(merged)
        context = RuntimeContext(workspace=workspace, **merged)
        with self._execution(context):
            try:
                await self._run_steps(context)
            except Exception as e:
                context.response.success = False
                context.response.answer = _describe_exception(e)
                await context.add_stream_string(str(e), ChunkEnum.ERROR)
            # Always emit DONE so consumers can detach even after an error.
            await context.add_stream_done()
