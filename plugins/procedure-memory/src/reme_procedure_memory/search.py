"""Shared procedure Job dependency and complete JSON responses."""

import json

from reme.schema import Response
from reme.steps.base_step import BaseStep, Ref

from .store import ProcedureMemoryStore


class ProcedureStep(BaseStep):
    """Resolve the application-owned store without shared state on Step instances."""

    procedure_memory = Ref(ProcedureMemoryStore, "procedure_memory")

    def respond(self, payload: dict, success: bool = True) -> Response:
        """AgentScope tool wrappers expose answer, so metadata alone is insufficient."""
        response = self.context.response
        response.success = success
        response.answer = json.dumps(payload, ensure_ascii=False)
        response.metadata.update(payload)
        return response


class ProcedureSearchStep(ProcedureStep):
    """Select either hybrid index; do not suppress previously seen experiences."""

    async def execute(self) -> Response:
        """Search the selected view and expose complete results to the tool caller."""
        return self.respond(
            await self.procedure_memory.search(
                self.context["query"],
                self.context.get("index", "when_to_use"),
                self.context.get("limit", 5),
                self.kwargs.get("vector_weight", 0.7),
            ),
        )
