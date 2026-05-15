"""``file_list`` — enumerate vault files under a directory.

Reads directly from the filesystem (``Path.iterdir`` /
``Path.rglob``), **not** the file_store index. The store may lag
behind disk during indexing or after rapid mutations; for the most
current view, the on-disk walk is the source of truth.

Parameters:
    path        — directory to list under (vault-relative or absolute).
                  Empty = working_dir root. Short links are not
                  meaningful for directories.
    limit       — cap the number of returned items.
    recursive   — descend into subdirectories. Default False = direct
                  children only.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum
from ...utils import path_resolver


def _walk(target_dir: Path, recursive: bool) -> Iterator[Path]:
    items = target_dir.rglob("*") if recursive else target_dir.iterdir()
    return (p for p in items if p.is_file())


def _list(
    file_store,
    *,
    path: str,
    recursive: bool,
    limit: int,
) -> dict:
    target_dir = path_resolver.to_absolute(file_store, path or ".")
    if not target_dir.is_dir():
        return {"items": [], "count": 0}
    working_dir = Path(getattr(file_store, "working_dir", None) or ".").resolve()
    items: list[dict] = []
    for entry in _walk(target_dir, recursive):
        try:
            rel = str(entry.relative_to(working_dir))
        except ValueError:
            rel = str(entry)
        items.append({"path": rel})
        if len(items) >= limit:
            break
    return {"items": items, "count": len(items)}


@R.register("list")
class FileList(BaseStep):
    """Enumerate vault files under a directory."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        result = _list(
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

        Reads the filesystem directly (no file_store cache). Returns
        ``{items: [{path}, ...], count}`` where ``path`` is
        vault-relative.

        Args:
            path: directory to list (relative to working_dir or absolute).
                Empty = working_dir root.
            limit: cap the number of returned items. Default 100.
            recursive: descend into subdirectories. Default False =
                direct children only.
        """
        result = _list(
            self.file_store,
            path=path,
            recursive=recursive,
            limit=limit,
        )
        return _tool_response("file_list", True, result, audit=self.audit)
