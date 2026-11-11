"""Explicit rebuild from experience JSON, never from stale derived chunks."""

from reme.schema import Response

from .search import ProcedureStep


class ProcedureReindexStep(ProcedureStep):
    """Report invalid files without rewriting or deleting them."""

    async def execute(self) -> Response:
        """Rebuild both views and return any invalid experience paths."""
        result = await self.procedure_memory.rebuild()
        return self.respond(result, success=not result["errors"])
