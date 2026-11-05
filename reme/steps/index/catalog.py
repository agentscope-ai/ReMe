"""List pending resources and acknowledge caller-processed packages."""

from ..base_step import BaseStep
from ...components import R


@R.register("resource_pending_step")
class ResourcePendingStep(BaseStep):
    """Return pending resource paths without updating the catalog."""

    async def execute(self):
        assert self.context is not None
        if self.file_catalog is None:
            raise RuntimeError("resource_pending_step requires file_catalog")
        self.context.response.success = True
        self.context.response.answer = await self.file_catalog.pending(limit=self.context.get("limit", 10))
        return self.context.response


@R.register("resource_ack_step")
class ResourceAckStep(BaseStep):
    """Acknowledge successfully processed resource paths as one batch."""

    async def execute(self):
        assert self.context is not None
        if self.file_catalog is None:
            raise RuntimeError("resource_ack_step requires file_catalog")
        result = await self.file_catalog.ack(self.context.get("paths"))
        self.context.response.answer = result
        self.context.response.success = result
        return self.context.response
