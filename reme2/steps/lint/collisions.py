"""``lint:collisions`` — find basenames resolving to >1 path.

Atomic vault-health check: groups every indexed path by its
filename + extension, surfaces any basename owned by more than one
path. Operates synchronously over the local file_store index — no
async graph round-trip needed.

Returns:

    {count: int, groups: {<basename>: [<path>, ...]}}

Each path list is sorted for stable output across runs.
"""

from __future__ import annotations

from pathlib import Path

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


def _scan_collisions(file_store) -> dict[str, list[str]]:
    by_name: dict[str, list[str]] = {}
    for path in file_store.file_nodes:
        by_name.setdefault(Path(path).name, []).append(path)
    return {name: sorted(paths) for name, paths in by_name.items() if len(paths) > 1}


@R.register("lint:collisions")
class LintCollisions(BaseStep):
    """List basenames resolving to >1 path (short-link ambiguity)."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        groups = _scan_collisions(self.file_store)
        _set_answer(self.context, {"count": len(groups), "groups": groups})

    async def lint_collisions(self) -> ToolResponse:
        """List basenames resolving to >1 path (short-link ambiguity)."""
        groups = _scan_collisions(self.file_store)
        return _tool_response(
            "lint:collisions", True,
            {"count": len(groups), "groups": groups},
            audit=self.audit,
        )
