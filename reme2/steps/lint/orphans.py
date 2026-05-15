"""``lint:orphans`` — find nodes with no inlinks AND no outlinks.

Atomic vault-health check: a node is an orphan if its ``links`` is
empty AND no other node references it. O(N + E) — one pass to mark
every referenced path, one pass to filter unreferenced + linkless
nodes.

Returns:

    {count: int, paths: [<path>, ...]}

The ``paths`` list is sorted for stable output across runs.
"""

from __future__ import annotations

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


def _scan_orphans(file_store) -> list[str]:
    nodes = file_store.file_nodes
    referenced: set[str] = set()
    for node in nodes.values():
        for link in node.links:
            if link.path:
                referenced.add(link.path)
    out: list[str] = []
    for path, node in nodes.items():
        has_out = any(link.path for link in node.links)
        has_in = path in referenced
        if not has_out and not has_in:
            out.append(path)
    return sorted(out)


@R.register("lint:orphans")
class LintOrphans(BaseStep):
    """List nodes with no inlinks AND no outlinks."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        orphans = _scan_orphans(self.file_store)
        _set_answer(self.context, {"count": len(orphans), "paths": orphans})

    async def lint_orphans(self) -> ToolResponse:
        """List nodes with no inlinks AND no outlinks."""
        orphans = _scan_orphans(self.file_store)
        return _tool_response(
            "lint:orphans", True,
            {"count": len(orphans), "paths": orphans},
            audit=self.audit,
        )
