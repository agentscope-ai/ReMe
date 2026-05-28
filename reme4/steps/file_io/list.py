"""``file_list`` — enumerate files under a directory in the vault.

Reads directly from the filesystem (``Path.iterdir`` / ``Path.rglob``),
**not** the file_store index. The store may lag behind disk during
indexing or after rapid mutations; the on-disk walk is the source of truth.

Parameters:
    path        — dir to list under (relative to the vault or absolute). Empty = vault root.
    limit       — cap on the number of returned items (default 100, must be > 0).
    recursive   — descend into subdirectories. Default False = direct children only.

No frontmatter is read. Callers needing frontmatter-based filtering
should iterate the result and call ``frontmatter_read`` per candidate.
"""

from pathlib import Path
from typing import Iterable

from ..base_step import BaseStep
from ...components import R

# Default cap on returned items so huge vaults don't blow up the response.
DEFAULT_LIMIT = 100


@R.register("list_step")
class ListStep(BaseStep):
    """Enumerate files under a directory in the vault."""

    def _fail(self, message: str, **meta) -> None:
        """Set a failed response (matches the read/edit/... fail envelope)."""
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    def _collect_params(self) -> tuple[str, bool, int]:
        """Read ``path`` / ``recursive`` / ``limit`` from context; coerce permissively."""
        assert self.context is not None
        path = str(self.context.get("path") or "")
        recursive = bool(self.context.get("recursive", False))
        raw_limit = self.context.get("limit")
        # Strings like "50" are accepted; bad/non-positive values fall back to default.
        try:
            limit = int(raw_limit) if raw_limit is not None else DEFAULT_LIMIT
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        return path, recursive, limit if limit > 0 else DEFAULT_LIMIT

    @staticmethod
    def _resolve_target_dir(vault_dir: Path, path: str) -> Path:
        """Empty → vault root; absolute → as-is; relative → joined under vault_dir."""
        if not path:
            return vault_dir
        candidate = Path(path)
        return candidate.resolve() if candidate.is_absolute() else (vault_dir / candidate).resolve()

    @staticmethod
    def _walk_files(target_dir: Path, recursive: bool, limit: int) -> list[Path]:
        """Return up to ``limit`` regular files under ``target_dir``; short-circuits at the cap."""
        entries: Iterable[Path] = target_dir.rglob("*") if recursive else target_dir.iterdir()
        files: list[Path] = []
        for entry in entries:
            if not entry.is_file():  # skip dirs, sockets, broken links, etc.
                continue
            files.append(entry)
            if len(files) >= limit:
                break
        return files

    @staticmethod
    def _format_relative(files: list[Path], vault_dir: Path) -> list[str]:
        """Render as vault-relative paths; fall back to absolute when outside the vault."""
        out: list[str] = []
        for entry in files:
            try:
                out.append(str(entry.relative_to(vault_dir)))
            except ValueError:
                out.append(str(entry))
        return out

    async def execute(self):
        assert self.context is not None
        path, recursive, limit = self._collect_params()
        vault_dir = Path(self.file_store.vault_path or ".").resolve()
        target_dir = self._resolve_target_dir(vault_dir, path)

        if not target_dir.exists():
            self._fail(f"directory {target_dir} does not exist", path=str(target_dir))
            return None
        if not target_dir.is_dir():
            self._fail(f"path {target_dir} is not a directory", path=str(target_dir))
            return None

        items = self._format_relative(self._walk_files(target_dir, recursive, limit), vault_dir)

        self.context.response.success = True
        self.context.response.answer = f"Listed {len(items)} file(s) under {path or '.'}"
        self.context.response.metadata.update({"items": items, "count": len(items)})
        self.logger.info(
            f"[{self.name}] listed dir={target_dir} recursive={recursive} count={len(items)} limit={limit}",
        )
        return self.context.response
