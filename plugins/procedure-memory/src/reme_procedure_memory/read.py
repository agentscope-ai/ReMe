"""Exact experience reads and bounded original-source excerpts."""

from reme.schema import Response

from .search import ProcedureStep
from .trajectory import prepare_text


class ProcedureReadStep(ProcedureStep):
    """Read only experiences or copied trajectories, never arbitrary host paths."""

    async def execute(self) -> Response:
        """Return a complete experience or an explicitly bounded source range."""
        store = self.procedure_memory
        relative = self.context["path"]
        if self.context.get("kind", "experience") == "experience":
            return self.respond(await store.read_experience(relative))
        if self.context.get("kind") != "trajectory":
            raise ValueError("kind must be experience or trajectory")
        path = store.source_path(relative)
        encoding = await prepare_text(path)
        offset = self.context.get("offset", 0)
        length = self.context.get("length", 8192)
        if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 50000:
            raise ValueError("offset must be nonnegative; length must be between 1 and 50000 characters")
        with path.open(encoding=encoding, newline="") as stream:
            remaining = offset
            while remaining:
                skipped = stream.read(min(remaining, 65536))
                if not skipped:
                    break
                remaining -= len(skipped)
            text = stream.read(length)
            end = store.budget.prefix(text, store.budget.limits["segment"]) if text else 0
            more = end < len(text) or bool(stream.read(1))
            text = text[:end]
        return self.respond(
            {"path": relative, "offset": offset, "text": text, "next_offset": offset + len(text), "more": more},
        )
