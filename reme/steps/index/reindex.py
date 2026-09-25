"""Explicit scoped rebuild of search indexes from already-ingested chunks."""

from ..base_step import BaseStep
from ...components import R


def _render_reindex(details: dict) -> str:
    """Render the store's reindex report as one readable summary line.

    ReMe Studio's settings panel shows ``response.answer`` when it is a string
    and otherwise falls back to a generic "index rebuilt" message
    (``reme_studio/app/settings-center.tsx``). Returning the report dict as the
    answer left that branch unreachable, so the operator never learned what was
    actually rebuilt. The report itself stays in ``metadata`` for programmatic
    consumers.
    """
    scope = str(details.get("scope") or "all")
    per_scope = {name: entry for name, entry in details.items() if isinstance(entry, dict)}
    if per_scope:
        counts = ", ".join(f"{name}={entry.get('indexed', 0)}" for name, entry in per_scope.items())
        return f"Reindexed scope={scope}: {counts}"
    return f"Reindexed scope={scope}: {details.get('indexed', 0)} indexed"


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
