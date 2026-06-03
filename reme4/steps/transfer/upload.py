"""``file_upload`` — copy a file from the local filesystem into the vault.

Symmetric counterpart to ``file_download``: source is on the local
filesystem, target is under the vault.

``src_path`` (filesystem source) is an absolute path on the host
filesystem (the file to copy in). Returns ``error="not found"``
when the file isn't on disk.

``dst_path`` is a path relative to the vault, **required**, and must
include a directory component so the caller is always explicit about
where in the vault the file lands. ``overwrite`` defaults to False —
callers must opt in to clobber an existing destination.

For the resource-bucket ingest path (channel-tagged, dated under
``resource/<YYYY-MM-DD>/`` with provenance metadata) use
``ingest`` instead.
"""

import mimetypes
import shutil
from pathlib import Path

from ..base_step import BaseStep

from ...components import R


@R.register("upload_step")
class UploadStep(BaseStep):
    """Copy ``src_path`` (on local fs) to ``dst_path`` (under the vault)."""

    async def execute(self):
        assert self.context is not None
        src_path: str = self.context.get("src_path", "") or ""
        dst_path: str = self.context.get("dst_path", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", False))
        assert src_path and dst_path, "src_path and dst_path are required"
        payload = await self._upload(src_path, dst_path, overwrite)
        if "error" in payload:
            self.context.response.success = False
            self.context.response.answer = f"Error: {payload['error']}"
            self.logger.info(
                f"[{self.name}] upload failed src={src_path} dst={dst_path} error={payload['error']!r}",
            )
        else:
            self.context.response.success = True
            self.context.response.answer = f"Uploaded {src_path} → {dst_path} ({payload['size']} bytes)"
            self.logger.info(
                f"[{self.name}] src={src_path} dst={dst_path} size={payload['size']} mime={payload['mime']}",
            )
        self.context.response.metadata.update(payload)

    async def _upload(self, src_path: str, dst_path: str, overwrite: bool) -> dict:
        src_abs = Path(src_path)
        if not src_abs.is_file():
            return {"src_path": src_path, "error": "not found"}
        if "/" not in dst_path:
            return {
                "dst_path": dst_path,
                "error": "dst_path must be relative to the vault with a directory component",
            }
        vault_dir = Path(self.file_store.vault_path or ".")
        dst_abs = (vault_dir / dst_path).resolve() if not Path(dst_path).is_absolute() else None
        if dst_abs is None:
            return {"dst_path": dst_path, "error": "dst_path must be relative to the vault"}
        if dst_abs.exists() and not overwrite:
            return {
                "src_path": src_path,
                "dst_path": dst_path,
                "error": "destination exists; pass overwrite=True",
            }
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_abs, dst_abs)
        return {
            "src_path": src_path,
            "dst_path": dst_path,
            "size": dst_abs.stat().st_size,
            "mime": mimetypes.guess_type(dst_abs.name)[0] or "application/octet-stream",
        }
