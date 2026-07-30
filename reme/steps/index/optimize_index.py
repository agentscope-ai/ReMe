"""Idle-time file store maintenance, scheduled off the request path (e.g. cron)."""

from ..base_step import BaseStep
from ...components import R


@R.register("optimize_index_step")
class OptimizeIndexStep(BaseStep):
    """Call ``file_store.optimize_index()`` so backends can compact derived index state."""

    async def execute(self):
        assert self.context is not None
        await self.file_store.optimize_index()
        self.context.response.metadata["optimized_index"] = True
        self.logger.info(f"[{self.name}] optimized file_store index")
        return self.context.response
