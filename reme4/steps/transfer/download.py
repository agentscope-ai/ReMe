"""``file_download`` — copy a file out of the vault to a local path.

Symmetric counterpart to ``file_upload``: source is under the vault,
target is on the local filesystem.

``src_path`` is a path relative to the vault (the file to copy out).
Returns ``error="not found"`` when the file isn't on disk.

``dst_path`` (filesystem target) is an absolute path on the host
filesystem. **Optional** — when empty, the file lands in a
session-scoped temp dir (lazy, auto-cleaned on process exit; each
call gets its own subdirectory so concurrent agents don't trample
each other) and the realized path is returned in ``dst_path``.
``overwrite`` defaults to False — callers must opt in to clobber an
existing destination.
"""

import mimetypes
import shutil
import tempfile
from pathlib import Path

from ..base_step import BaseStep

from ...components import R


_TEMP_ROOT: Path | None = None


def _get_temp_root() -> Path:
    """Lazy session-scoped temp dir. Auto-cleaned on process exit."""
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _TEMP_ROOT = Path(tempfile.mkdtemp(prefix="reme-files-"))
    return _TEMP_ROOT


@R.register("download_step")
class DownloadStep(BaseStep):
    """Copy ``src_path`` (under the vault) to ``dst_path`` (or a temp file if omitted)."""

    async def execute(self):
        assert self.context is not None
        src_path: str = self.context.get("src_path", "") or ""
        dst_path: str = self.context.get("dst_path", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", False))
        assert src_path, "src_path is required"
        payload = await self._download(src_path, dst_path, overwrite)
        if "error" in payload:
            self.context.response.success = False
            self.context.response.answer = f"Error: {payload['error']}"
            self.logger.info(f"[{self.name}] download failed src={src_path} error={payload['error']!r}")
        else:
            self.context.response.success = True
            self.context.response.answer = f"Downloaded {src_path} → {payload['dst_path']} ({payload['size']} bytes)"
            self.logger.info(
                f"[{self.name}] src={src_path} dst={payload['dst_path']} "
                f"size={payload['size']} mime={payload['mime']}",
            )
        self.context.response.metadata.update(payload)

    async def _download(self, src_path: str, dst_path: str, overwrite: bool) -> dict:
        if not src_path:
            return {"src_path": src_path, "error": "not found"}
        src_abs = (Path(self.file_store.vault_path or ".") / src_path).resolve()
        if not src_abs.is_file():
            return {"src_path": src_path, "error": "not found"}

        if dst_path:
            dst_abs = Path(dst_path)
            if dst_abs.exists() and not overwrite:
                return {
                    "src_path": src_path,
                    "dst_path": dst_path,
                    "error": "destination exists; pass overwrite=True",
                }
            dst_abs.parent.mkdir(parents=True, exist_ok=True)
        else:
            dst_dir = Path(tempfile.mkdtemp(prefix="dl-", dir=_get_temp_root()))
            dst_abs = dst_dir / src_abs.name

        shutil.copy2(src_abs, dst_abs)
        return {
            "src_path": src_path,
            "dst_path": str(dst_abs),
            "size": dst_abs.stat().st_size,
            "mime": mimetypes.guess_type(dst_abs.name)[0] or "application/octet-stream",
        }
