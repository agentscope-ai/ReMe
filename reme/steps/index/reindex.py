"""Explicit scoped rebuild of search indexes from already-ingested chunks."""

from ..base_step import BaseStep
from ...core.registry import R


def _render_entry(name: str, entry: dict) -> str:
    """Render one index report, flagging a degraded optional index."""
    rendered = f"{name}={entry.get('indexed', 0)}"
    return f"{rendered} (failed)" if entry.get("failed") else rendered


def _render_reindex(details: dict) -> str:
    """Render the store's reindex report as one readable summary line."""
    scope = str(details.get("scope") or "all")
    counts = ", ".join(_render_entry(name, entry) for name, entry in details.items() if isinstance(entry, dict))
    if counts:
        summary = counts
    else:
        # Flat single-index report: the failure marker sits at the top level, so
        # it has to be rendered here or it disappears from ``answer`` -- which is
        # all that MCP callers and the default CLI output ever look at.
        summary = f"{details.get('indexed', 0)} indexed"
        if details.get("failed"):
            summary += " (failed)"
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
