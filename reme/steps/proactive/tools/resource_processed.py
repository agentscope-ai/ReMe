"""Acknowledge successfully processed resource packages."""

from ...base_step import BaseStep
from ....components import R


@R.register("resource_processed_step")
class ResourceProcessedStep(BaseStep):
    """Mark successfully processed resource paths as one batch."""

    async def execute(self):
        assert self.context is not None
        if self.file_catalog is None:
            raise RuntimeError("resource_processed_step requires file_catalog")
        result = await self.file_catalog.processed(self.context.get("paths"))
        self.context.response.answer = result
        self.context.response.success = result
        return self.context.response
