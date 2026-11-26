"""Common helpers for workspace file operations."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from ._path import check_path_permission, display_path, gate_md, resolve_path
from ..base_step import BaseStep
from ...utils.files.path_lock import get_path_lock


@asynccontextmanager
async def frontmatter_target(
    workspace_path: Path,
    path: str,
    allowed_paths: object,
) -> AsyncIterator[tuple[Path | None, dict]]:
    """Yield a locked Markdown target or its path error payload."""
    workspace = Path(workspace_path or ".").resolve()
    target, error = resolve_path(workspace, path)
    payload = {"path": path}
    if error or target is None:
        payload["error"] = error or "invalid path"
        yield None, payload
        return

    original_target = target
    target, is_md = gate_md(target)
    if not check_path_permission(workspace, target, allowed_paths):
        payload["error"] = "no permission to update this file"
        yield None, payload
    else:
        async with await get_path_lock(target):
            if not target.is_file():
                payload["error"] = f"{display_path(workspace, target)} not found"
            elif not is_md:
                payload["error"] = "not markdown"
            yield None if "error" in payload else target, payload
    if target != original_target:
        payload["resolved_path"] = display_path(workspace, target)


class _FileIOStep(BaseStep):
    """Preserve the file-operation failure envelope without changing its payload."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        self.context.response.metadata.update(meta)
