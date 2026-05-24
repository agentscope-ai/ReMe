"""``file_download`` — copy a file out of the vault to a local path.

Symmetric counterpart to ``file_upload``: source is under the vault,
target is on the local filesystem.

``path`` is a path relative to the vault (the file to copy out).
Returns ``error="not found"`` when the file isn't on disk.

``download_path`` (filesystem target) is an absolute path on the host
filesystem. **Optional** — when empty, the file lands in a
session-scoped temp dir (lazy, auto-cleaned on process exit; each
call gets its own subdirectory so concurrent agents don't trample
each other) and the realized path is returned in ``download_path``.
``overwrite=True`` is the default since most download flows are
intentional refreshes.
"""

from __future__ import annotations

import mimetypes
import shutil
import tempfile
from pathlib import Path

from ..base_step import BaseStep
from ...utils import set_answer

from ...components import R
from ...enumeration import ComponentEnum


_TEMP_ROOT: Path | None = None


def _get_temp_root() -> Path:
    """Lazy session-scoped temp dir. Auto-cleaned on process exit."""
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _TEMP_ROOT = Path(tempfile.mkdtemp(prefix="reme2-files-"))
    return _TEMP_ROOT


@R.register("download_step")
class DownloadStep(BaseStep):
    """Copy ``path`` (under the vault) to ``download_path`` (or a temp file if omitted)."""

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        download_path: str = self.context.get("download_path", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", True))
        assert path, "path is required"
        payload = await self._download(path, download_path, overwrite)
        self.context.response.success = "error" not in payload
        set_answer(self.context, payload)

    async def _download(self, path: str, download_path: str, overwrite: bool) -> dict:
        if not path:
            return {"path": path, "error": "not found"}
        src_path = (Path(self.file_store.vault_path or ".") / path).resolve()
        if not src_path.is_file():
            return {"path": path, "error": "not found"}

        if download_path:
            dst_path = Path(download_path)
            if dst_path.exists() and not overwrite:
                return {
                    "path": path,
                    "download_path": download_path,
                    "error": "destination exists; pass overwrite=True",
                }
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            dst_dir = Path(tempfile.mkdtemp(prefix="dl-", dir=_get_temp_root()))
            dst_path = dst_dir / src_path.name

        shutil.copy2(src_path, dst_path)
        return {
            "path": path,
            "download_path": str(dst_path),
            "size": dst_path.stat().st_size,
            "mime": mimetypes.guess_type(dst_path.name)[0] or "application/octet-stream",
        }
