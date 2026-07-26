"""Idle-time file store maintenance, scheduled off the request path (e.g. cron)."""

from ..base_step import BaseStep
from ...components import R


@R.register("refine_store_step")
class RefineStoreStep(BaseStep):
    """Call ``file_store.refine()`` so backends can compact derived index state."""

    async def execute(self):
        assert self.context is not None
        await self.file_store.refine()
        self.context.response.metadata["refined_store"] = True
        self.logger.info(f"[{self.name}] refined file_store")
        return self.context.response
