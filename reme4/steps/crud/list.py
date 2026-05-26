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

from pathlib import Path

from ..base_step import BaseStep

from ...components import R


@R.register("list_step")
class ListStep(BaseStep):
    """Enumerate files under a directory in the vault."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        recursive: bool = bool(self.context.get("recursive", False))
        limit: int = int(self.context.get("limit") or 100)

        vault_dir = Path(self.file_store.vault_path or ".").resolve()
        target_dir = (vault_dir / (path or ".")).resolve()
        items: list[str] = []
        if target_dir.is_dir():
            entries = target_dir.rglob("*") if recursive else target_dir.iterdir()
            for entry in entries:
                if not entry.is_file():
                    continue
                try:
                    rel = str(entry.relative_to(vault_dir))
                except ValueError:
                    rel = str(entry)
                items.append(rel)
                if len(items) >= limit:
                    break

        self.context.response.success = True
        self.context.response.answer = f"Listed {len(items)} file(s) under {path or '.'}"
        self.context.response.metadata.update({"items": items, "count": len(items)})
