"""Public catalog diff and commit operations; business processing stays with the caller."""

from ..base_step import BaseStep
from ...components import R


@R.register("catalog_diff_step")
class CatalogDiffStep(BaseStep):
    """Return pending file changes without updating the catalog."""

    async def execute(self):
        assert self.context is not None
        if self.file_catalog is None:
            raise RuntimeError("catalog_diff_step requires file_catalog")
        result = await self.file_catalog.diff()
        self.context["changes"] = result["changes"]
        self.context.response.success = True
        self.context.response.answer = result
        self.context.response.metadata.update(result)
        return self.context.response


@R.register("catalog_commit_step")
class CatalogCommitStep(BaseStep):
    """Confirm only caller-selected changes whose versions remain current."""

    async def execute(self):
        assert self.context is not None
        if self.file_catalog is None:
            raise RuntimeError("catalog_commit_step requires file_catalog")
        changes = self.context.get("changes")
        if not isinstance(changes, list):
            raise ValueError("catalog_commit requires changes: list")
        result = await self.file_catalog.commit(changes)
        self.context.response.answer = result
        self.context.response.metadata.update(result)
        self.context.response.success = not result["skipped"]
        return self.context.response
