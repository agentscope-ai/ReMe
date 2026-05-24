"""``file_list`` — enumerate files under a directory in the vault.

Reads directly from the filesystem (``Path.iterdir`` /
``Path.rglob``), **not** the file_store index. The store may lag
behind disk during indexing or after rapid mutations; for the most
current view, the on-disk walk is the source of truth.

Parameters:
    path        — directory to list under (relative to the vault or absolute).
                  Empty = vault root.
    limit       — cap the number of returned items.
    recursive   — descend into subdirectories. Default False = direct
                  children only.

No frontmatter is read — this is a plain directory walker. Callers
that need frontmatter-based filtering should iterate the result and
call ``frontmatter_read`` per candidate.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..base_step import BaseStep
from ...utils import set_answer

from ...components import R
from ...enumeration import ComponentEnum


def _walk(target_dir: Path, recursive: bool) -> Iterator[Path]:
    items = target_dir.rglob("*") if recursive else target_dir.iterdir()
    return (p for p in items if p.is_file())


async def _list(
    file_store,
    *,
    path: str,
    recursive: bool,
    limit: int,
) -> dict:
    vault_dir = Path(file_store.vault_path or ".").resolve()
    target_dir = (vault_dir / (path or ".")).resolve()
    if not target_dir.is_dir():
        return {"items": [], "count": 0}
    items: list[str] = []
    for entry in _walk(target_dir, recursive):
        try:
            rel = str(entry.relative_to(vault_dir))
        except ValueError:
            rel = str(entry)
        items.append(rel)
        if len(items) >= limit:
            break
    return {"items": items, "count": len(items)}


@R.register("list_step")
class ListStep(BaseStep):
    """Enumerate files under a directory in the vault."""

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        result = await _list(
            self.file_store,
            path=self.context.get("path") or "",
            recursive=bool(self.context.get("recursive", False)),
            limit=int(self.context.get("limit") or 100),
        )
        set_answer(self.context, result)
