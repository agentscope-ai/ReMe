"""``file_download`` — copy a vault file to a session temp dir.

Agent flow: agent calls ``file_download(path)``, gets back a
local path under a fresh per-call temp directory, then opens / parses
the file with whatever tooling it likes. The vault copy is untouched.

The temp root is lazy and session-scoped — created on first download,
left for the OS to clean up at process exit. Each download lands in
its own subdirectory so concurrent agents don't trample each other.

``path`` accepts a short link or a vault-relative path; resolution
goes through ``path_resolver.resolve_to_absolute``. Short-link
ambiguity is surfaced as ``error="ambiguous"`` with the candidate
list — the download is **not** executed in that case.
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
from ...utils import path_resolver


_TEMP_ROOT: Path | None = None


def _get_temp_root() -> Path:
    """Lazy session-scoped temp dir. Auto-cleaned on process exit."""
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _TEMP_ROOT = Path(tempfile.mkdtemp(prefix="reme2-files-"))
    return _TEMP_ROOT


@R.register("file_download")
class FileDownload(BaseStep):
    """Copy a vault file to a session temp dir; return the local path."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"
        payload = await self._download(path)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def file_download(self, path: str) -> ToolResponse:
        """Copy a vault file to session temp dir; return the local path."""
        payload = await self._download(path)
        ok = "error" not in payload
        return _tool_response("file_download", ok, payload, audit=self.audit)

    async def _download(self, path: str) -> dict:
        try:
            src = await path_resolver.resolve_to_absolute(self.file_store, path)
        except path_resolver.PathAmbiguous as e:
            return {"path": path, "error": "ambiguous", "candidates": e.candidates}
        except path_resolver.PathNotFound:
            return {"path": path, "error": "not found"}
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
