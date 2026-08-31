"""Explicit scoped rebuild of search indexes from already-ingested chunks."""

from ..base_step import BaseStep
from ...components import R


@R.register("reindex_step")
class ReindexStep(BaseStep):
    """Rebuild BM25 and/or embeddings without scanning files or changing the graph."""

    async def execute(self):
        assert self.context is not None
        scope = str(self.context.get("scope", "all"))
        details = await self.file_store.reindex(scope)

        self.context.response.answer = details
        self.context.response.metadata.update(details)
        self.context.response.metadata["scope"] = scope
        return self.context.response
