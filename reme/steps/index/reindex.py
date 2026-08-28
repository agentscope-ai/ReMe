"""Explicit scoped search-index rebuild."""

from ..base_step import BaseStep
from ...components import R


@R.register("reindex_step")
class ReindexStep(BaseStep):
    """Rebuild ``all``, ``bm25``, or ``embedding`` synchronously."""

    async def execute(self):
        assert self.context is not None
        scope = str(self.context.get("scope", "all"))
        details = await self.file_store.reindex(scope)

        self.context.response.answer = details
        self.context.response.metadata.update(details)
        self.context.response.metadata["scope"] = scope
        return self.context.response
