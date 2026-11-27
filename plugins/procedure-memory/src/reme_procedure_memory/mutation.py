"""Controlled experience writes, usable only inside an active extraction run."""

from reme.schema import Response

from .search import ProcedureStep
from .store import ProcedureIndexError


class ProcedureCreateStep(ProcedureStep):
    """Create one semantically named experience and synchronize both indexes."""

    async def execute(self) -> Response:
        """Apply a create operation within the server-injected scope."""
        arguments = {
            key: self.context[key] for key in ("filename", "description", "when-to-use", "content", "operation_key")
        }
        try:
            result = await self.procedure_memory.apply_mutation(
                "create",
                arguments,
                self.context.get("_procedure_scope", ""),
            )
        except ProcedureIndexError as exc:
            return self.respond({**exc.result, "repair": str(exc)}, success=False)
        return self.respond(result)


class ProcedureUpdateStep(ProcedureStep):
    """Rewrite selected text fields with a version check and preserved sources."""

    async def execute(self) -> Response:
        """Apply a version-checked update within the server-injected scope."""
        arguments = {key: self.context[key] for key in ("path", "expected_version", "changes", "operation_key")}
        try:
            result = await self.procedure_memory.apply_mutation(
                "update",
                arguments,
                self.context.get("_procedure_scope", ""),
            )
        except ProcedureIndexError as exc:
            return self.respond({**exc.result, "repair": str(exc)}, success=False)
        return self.respond(result)
