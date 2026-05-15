"""``file_list`` — enumerate vault files under a directory.

Walks the file_graph in one shot via ``get_nodes(None)`` — no
filesystem scan, no per-file frontmatter parse, no per-file graph
round-trip.

Parameters:
    path        — directory to list under (vault-relative or absolute).
                  Empty = working_dir root. Short links are not
                  meaningful for directories.
    limit       — cap the number of returned items.
    recursive   — descend into subdirectories. Default False = direct
                  children only.
"""

from __future__ import annotations

from pathlib import Path

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum
from ...utils import path_resolver


def _under(node_abs: Path, target_dir: Path, *, recursive: bool) -> bool:
    if recursive:
        return node_abs == target_dir or target_dir in node_abs.parents
    return node_abs.parent == target_dir


async def _list(
    file_store,
    *,
    path: str,
    recursive: bool,
    limit: int,
) -> dict:
    if not file_store.file_graph:
        return {"items": [], "count": 0}
    target_dir = path_resolver.to_absolute(file_store, path or ".")
    items: list[dict] = []
    for node in await file_store.file_graph.get_nodes():
        node_abs = path_resolver.to_absolute(file_store, node.path)
        if not _under(node_abs, target_dir, recursive=recursive):
            continue
        items.append({"path": node.path, "metadata": node.front_matter.model_dump()})
        if len(items) >= limit:
            break
    return {"items": items, "count": len(items)}


@R.register("file_list")
class FileList(BaseStep):
    """Enumerate vault files under a directory."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        result = await _list(
            self.file_store,
            path=self.context.get("path") or "",
            recursive=bool(self.context.get("recursive", False)),
            limit=int(self.context.get("limit") or 100),
        )
        _set_answer(self.context, result)

    async def file_list(
        self,
        path: str = "",
        limit: int = 100,
        recursive: bool = False,
    ) -> ToolResponse:
        """List vault files under ``path``.

        Args:
            path: directory to list (relative to working_dir or absolute).
                Empty = working_dir root.
            limit: cap the number of returned items. Default 100.
            recursive: descend into subdirectories. Default False =
                direct children only.
        """
        result = await _list(
            self.file_store,
            path=path,
            recursive=recursive,
            limit=limit,
        )
        return _tool_response("file_list", True, result, audit=self.audit)
