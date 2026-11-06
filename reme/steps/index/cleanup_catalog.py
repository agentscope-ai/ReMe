"""Explicit retention cleanup for dated resource directories."""

from ..base_step import BaseStep
from ...components import R


@R.register("cleanup_catalog_step")
class CleanupCatalogStep(BaseStep):
    """Ask a file catalog to remove expired directories under ``resource_dir``."""

    async def execute(self):
        assert self.context is not None
        if self.file_catalog is None:
            raise RuntimeError("cleanup_catalog_step requires file_catalog")
        result = await self.file_catalog.cleanup_expired()
        self.context.response.metadata.update(result)
        self.context.response.success = not result["errors"]
        self.context.response.answer = f"Deleted {len(result['deleted_dirs'])} expired resource directories"
        if result["errors"]:
            self.context.response.answer += f"; failed to delete {len(result['errors'])} directories"
        return self.context.response
