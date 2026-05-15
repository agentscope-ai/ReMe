"""``file_download`` — copy a vault file to a session temp dir.

Agent flow: agent calls ``file_download(path)``, gets back a
local path under a fresh per-call temp directory, then opens / parses
the file with whatever tooling it likes. The vault copy is untouched.

The temp root is lazy and session-scoped — created on first download,
left for the OS to clean up at process exit. Each download lands in
its own subdirectory so concurrent agents don't trample each other.

Also exports ``resolve_path`` — the shared helper for turning
a vault-relative or absolute path into an absolute on-disk path.
``upload`` and ``list`` import it from here.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


_TEMP_ROOT: Path | None = None


def _get_temp_root() -> Path:
    """Lazy session-scoped temp dir. Auto-cleaned on process exit."""
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _TEMP_ROOT = Path(tempfile.mkdtemp(prefix="reme2-files-"))
    return _TEMP_ROOT


def resolve_path(file_store, path: str) -> Path:
    """Compose the absolute on-disk path for relative entry.

    Public so sibling steps (``upload``, ``list``, event tools) can
    reuse the same path-resolution rule. Absolute paths pass through;
    relative paths join under ``file_store.working_dir``.
    """
    working_dir = getattr(file_store, "working_dir", None) or "."
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (Path(working_dir) / p).resolve()


@R.register("file_download")
class FileDownload(BaseStep):
    """Copy a vault file to a session temp dir; return the local path."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"
        payload = self._download(path)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def file_download(self, path: str) -> ToolResponse:
        """Copy a vault file to session temp dir; return the local path."""
        payload = self._download(path)
        ok = "error" not in payload
        return _tool_response("file_download", ok, payload, audit=self.audit)

    def _download(self, path: str) -> dict:
        src = resolve_path(self.file_store, path)
        if not src.is_file():
            return {"path": path, "error": "not found"}
        dst_dir = Path(tempfile.mkdtemp(prefix="dl-", dir=_get_temp_root()))
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return {
            "path": path,
            "local_path": str(dst),
            "size": dst.stat().st_size,
        }
