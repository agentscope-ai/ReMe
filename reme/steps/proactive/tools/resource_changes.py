"""List resource packages awaiting processing."""

from ...base_step import BaseStep
from ....components import R


@R.register("resource_changes_step")
class ResourceChangesStep(BaseStep):
    """Return new or modified resource paths awaiting processing without updating the catalog."""

    async def execute(self):
        assert self.context is not None
        if self.file_catalog is None:
            raise RuntimeError("resource_changes_step requires file_catalog")
        self.context.response.success = True
        self.context.response.answer = await self.file_catalog.changes(limit=self.context.get("limit", 20))
        return self.context.response
