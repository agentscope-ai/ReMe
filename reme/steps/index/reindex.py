"""Explicit scoped rebuild of search indexes from already-ingested chunks."""

from ..base_step import BaseStep
from ...core.registry import R


def _render_reindex(details: dict) -> str:
    """Render the store's reindex report as one readable summary line."""
    scope = str(details.get("scope") or "all")
    counts = ", ".join(
        f"{name}={entry.get('indexed', 0)}" for name, entry in details.items() if isinstance(entry, dict)
    )
    summary = counts or f"{details.get('indexed', 0)} indexed"
    return f"Reindexed scope={scope}: {summary}"


@R.register("reindex_step")
class ReindexStep(BaseStep):
    """Rebuild BM25, embeddings, and/or tags without scanning workspace files."""

    async def execute(self):
        assert self.context is not None
        scope = str(self.context.get("scope", "all"))
        details = await self.file_store.reindex(scope)

        self.context.response.answer = _render_reindex(details)
        self.context.response.metadata.update(details)
        self.context.response.metadata["scope"] = scope
        return self.context.response
