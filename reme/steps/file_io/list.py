"""``file_list`` — enumerate files under a directory in the workspace.

Reads directly from the filesystem (``Path.iterdir`` / ``Path.rglob``),
**not** the file_store index. The store may lag behind disk during
indexing or after rapid mutations; the on-disk walk is the source of truth.

Parameters:
    path        — dir to list under (relative to the workspace or absolute). Empty = workspace root.
    limit       — cap on the number of returned items (default 100, must be > 0).
    recursive   — descend into subdirectories. Default False = direct children only.
    sort_by     — optional ordering; ``mtime`` returns most recently modified files first.
    extensions  — optional extension allowlist applied before sorting and limiting.

No frontmatter is read. Callers needing frontmatter-based filtering
should iterate the result and call ``frontmatter_read`` per candidate.
"""

from pathlib import Path
from stat import S_ISREG
from typing import Iterable

from ._path import resolve_path
from ..base_step import BaseStep
from ...components import R

# Default cap on returned items so huge workspaces don't blow up the response.
DEFAULT_LIMIT = 100


@R.register("list_step")
class ListStep(BaseStep):
    """Enumerate files under a directory in the workspace."""

    def _fail(self, message: str, **meta) -> None:
        """Set a failed response (matches the read/edit/... fail envelope)."""
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    def _collect_params(self) -> tuple[str, bool, int, str, frozenset[str]]:
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
        sort_by = str(self.context.get("sort_by") or "")
        raw_extensions = self.context.get("extensions") or []
        if isinstance(raw_extensions, str):
            raw_extensions = raw_extensions.split(",")
        extensions = frozenset(
            normalized for value in raw_extensions if (normalized := str(value).strip().lower().lstrip("."))
        )
        return path, recursive, limit if limit > 0 else DEFAULT_LIMIT, sort_by, extensions

    @staticmethod
    def _walk_files(
        target_dir: Path,
        recursive: bool,
        limit: int,
        sort_by: str = "",
        extensions: frozenset[str] = frozenset(),
    ) -> list[Path]:
        """Return up to ``limit`` regular files, scanning all entries only when sorting."""
        entries: Iterable[Path] = target_dir.rglob("*") if recursive else target_dir.iterdir()
        files: list[Path] = []
        mtimes: dict[Path, int] = {}
        for entry in entries:
            if sort_by == "mtime":
                try:
                    entry_stat = entry.stat()
                except OSError:
                    continue
                if not S_ISREG(entry_stat.st_mode):
                    continue
            elif not entry.is_file():  # skip dirs, sockets, broken links, etc.
                continue
            if extensions and entry.suffix.lower().lstrip(".") not in extensions:
                continue
            files.append(entry)
            if sort_by == "mtime":
                mtimes[entry] = entry_stat.st_mtime_ns
            elif len(files) >= limit:
                break
        if sort_by == "mtime":
            files.sort(key=lambda entry: (-mtimes[entry], entry.as_posix()))
        return files[:limit]

    @staticmethod
    def _format_relative(files: list[Path], workspace_dir: Path) -> list[str]:
        """Render as workspace-relative paths; fall back to absolute when outside the workspace."""
        out: list[str] = []
        for entry in files:
            try:
                out.append(entry.relative_to(workspace_dir).as_posix())
            except ValueError:
                out.append(str(entry))
        return out

    async def execute(self):
        assert self.context is not None
        path, recursive, limit, sort_by, extensions = self._collect_params()
        workspace_dir = Path(self.file_store.workspace_path or ".").resolve()
        target_dir, err = resolve_path(workspace_dir, path, allow_empty=True)
        if err or target_dir is None:
            self._fail(err or "invalid path", path=path)
            return None

        if not target_dir.exists():
            self._fail(f"directory {target_dir} does not exist", path=str(target_dir))
            return None
        if not target_dir.is_dir():
            self._fail(f"path {target_dir} is not a directory", path=str(target_dir))
            return None

        items = self._format_relative(
            self._walk_files(target_dir, recursive, limit, sort_by, extensions),
            workspace_dir,
        )

        self.context.response.success = True
        location = path or "."
        if items:
            rendered_items = "\n".join(f"- {item}" for item in items)
            self.context.response.answer = f"Listed {len(items)} file(s) under {location}:\n{rendered_items}"
        else:
            self.context.response.answer = f"No files found under {location}."
        self.context.response.metadata.update({"items": items, "count": len(items)})
        self.logger.info(
            f"[{self.name}] listed dir={target_dir} recursive={recursive} count={len(items)} limit={limit}",
        )
        return self.context.response
