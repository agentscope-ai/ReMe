"""``lint:dangling`` — find FileLinks pointing to non-existent nodes.

Atomic vault-health check: walks every indexed node's ``links`` and
reports each edge whose ``path`` target isn't present in the index.
Pure dict iteration over ``file_store.file_nodes`` — no filesystem
walk, no graph round-trip.

Returns:

    {count: int, findings: [
        {source, target, predicate, anchor},
        ...
    ]}
"""

from __future__ import annotations

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


def _scan_dangling(file_store) -> list[dict]:
    """One entry per dangling edge."""
    nodes = file_store.file_nodes
    out: list[dict] = []
    for src_path, node in nodes.items():
        for link in node.links:
            if not link.path:
                continue
            if link.path not in nodes:
                out.append({
                    "source": src_path,
                    "target": link.path,
                    "predicate": link.predicate,
                    "anchor": link.anchor,
                })
    return out


@R.register("lint:dangling")
class LintDangling(BaseStep):
    """List every FileLink whose target is not in the graph."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        findings = _scan_dangling(self.file_store)
        _set_answer(self.context, {"count": len(findings), "findings": findings})

    async def lint_dangling(self) -> ToolResponse:
        """List every FileLink whose target is not in the graph."""
        findings = _scan_dangling(self.file_store)
        return _tool_response(
            "lint:dangling", True,
            {"count": len(findings), "findings": findings},
            audit=self.audit,
        )
