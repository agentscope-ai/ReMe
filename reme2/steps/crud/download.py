"""``file_download`` — copy a vault file to a local path.

Symmetric counterpart to ``file_upload``: source is in the vault,
target is on the local filesystem.

``path`` (source) accepts a short link or vault-relative path;
resolution goes through ``path_resolver.resolve_to_absolute``.
Short-link ambiguity is surfaced as ``error="ambiguous"`` with the
candidate list — the download is **not** executed in that case.

``local_path`` (target) is a plain filesystem path. **Optional** —
when empty, the file lands in a session-scoped temp dir (lazy,
auto-cleaned on process exit; each call gets its own subdirectory
so concurrent agents don't trample each other). ``overwrite=True``
is the default since most download flows are intentional refreshes.
"""

from __future__ import annotations

import mimetypes
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


@R.register("download")
class FileDownload(BaseStep):
    """Copy a vault file to ``local_path`` (or a temp dir if omitted)."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        local_path: str = self.context.get("local_path", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", True))
        assert path, "path is required"
        payload = await self._download(path, local_path, overwrite)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def file_download(
        self, path: str, local_path: str = "", overwrite: bool = True,
    ) -> ToolResponse:
        """Copy the vault file at ``path`` to ``local_path``.

        ``local_path`` empty → land in a fresh temp subdirectory under
        the session temp root.
        """
        payload = await self._download(path, local_path, overwrite)
        ok = "error" not in payload
        return _tool_response("file_download", ok, payload, audit=self.audit)

    async def _download(self, path: str, local_path: str, overwrite: bool) -> dict:
        try:
            src = await path_resolver.resolve_to_absolute(self.file_store, path)
        except path_resolver.PathAmbiguous as e:
            return {"path": path, "error": "ambiguous", "candidates": e.candidates}
        except path_resolver.PathNotFound:
            return {"path": path, "error": "not found"}
        if not src.is_file():
            return {"path": path, "error": "not found"}

        if local_path:
            dst = Path(local_path)
            if dst.exists() and not overwrite:
                return {"local_path": local_path, "error": "destination exists; pass overwrite=True"}
            dst.parent.mkdir(parents=True, exist_ok=True)
        else:
            dst_dir = Path(tempfile.mkdtemp(prefix="dl-", dir=_get_temp_root()))
            dst = dst_dir / src.name

        shutil.copy2(src, dst)
        return {
            "path": path,
            "local_path": str(dst),
            "size": dst.stat().st_size,
            "mime": mimetypes.guess_type(dst.name)[0] or "application/octet-stream",
        }
