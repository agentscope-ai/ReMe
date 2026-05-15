"""``file_list`` — enumerate vault files with optional frontmatter filters.

Walks the file_graph in one shot via ``get_nodes(None)`` — no filesystem
scan, no per-file frontmatter parse, no per-file graph round-trip.

Filters compose:
    path_prefix    — prefix match against the indexed path string
    tags           — every requested tag must be present in ``tags``
    metadata       — frontmatter must equal each ``{key: value}`` pair

The ``metadata`` filter sees the full frontmatter dict — typed fields
(title / description / tags) merged with any ``extra=allow`` extras —
so agents can filter on schema-known keys or arbitrary extras alike.
"""

from __future__ import annotations

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


def _matches(
    path: str,
    md: dict,
    *,
    path_prefix: str | None,
    tags: list[str],
    metadata: dict,
) -> bool:
    if path_prefix and not path.startswith(path_prefix):
        return False
    if metadata and any(md.get(k) != v for k, v in metadata.items()):
        return False
    if tags:
        file_tags = set(md.get("tags") or [])
        if not all(t in file_tags for t in tags):
            return False
    return True


async def _list(
    file_store,
    *,
    path_prefix: str | None,
    tags: list[str],
    metadata: dict,
    limit: int,
) -> dict:
    if not file_store.file_graph:
        return {"items": [], "count": 0}
    items: list[dict] = []
    for node in await file_store.file_graph.get_nodes():
        md = node.front_matter.model_dump()
        if not _matches(node.path, md, path_prefix=path_prefix, tags=tags, metadata=metadata):
            continue
        items.append({"path": node.path, "metadata": md})
        if len(items) >= limit:
            break
    return {"items": items, "count": len(items)}


@R.register("file_list")
class FileList(BaseStep):
    """Enumerate vault files with optional frontmatter filters."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        result = await _list(
            self.file_store,
            path_prefix=self.context.get("prefix") or self.context.get("path_prefix"),
            tags=self.context.get("tags") or [],
            metadata=self.context.get("metadata") or {},
            limit=int(self.context.get("limit") or 100),
        )
        _set_answer(self.context, result)

    async def file_list(
        self,
        prefix: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        limit: int = 100,
    ) -> ToolResponse:
        """List vault files. Filters: path prefix, frontmatter tags / fields."""
        result = await _list(
            self.file_store,
            path_prefix=prefix,
            tags=tags or [],
            metadata=metadata or {},
            limit=limit,
        )
        return _tool_response("file_list", True, result, audit=self.audit)
