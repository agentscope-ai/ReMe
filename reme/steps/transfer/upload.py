"""``file_upload`` — copy a file from the local filesystem into the workspace.

Symmetric counterpart to ``file_download``: source is on the local
filesystem, target is under the workspace.

``src_path`` (filesystem source) is an absolute path on the host
filesystem (the file to copy in). Returns ``error="not found"``
when the file isn't on disk.

``dst_path`` is a path relative to the workspace, **required**, and must
include a directory component so the caller is always explicit about
where in the workspace the file lands. ``overwrite`` defaults to False —
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
    """Copy ``src_path`` (on local fs) to ``dst_path`` (under the workspace)."""

    async def execute(self):
        assert self.context is not None
        src_path: str = self.context.get("src_path", "") or ""
        dst_path: str = self.context.get("dst_path", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", False))
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

    async def _upload(
        self,
        src_path: str,
        dst_path: str,
        overwrite: bool,
    ) -> dict:  # pylint: disable=too-many-return-statements
        # pylint: disable=too-many-return-statements
        if not src_path:
            return {"src_path": src_path, "error": "src_path is required"}
        if not dst_path:
            return {"dst_path": dst_path, "error": "dst_path is required"}

        src_abs = Path(src_path).resolve()
        if not src_abs.is_file():
            return {"src_path": src_path, "error": "not found"}
        if "/" not in dst_path:
            return {
                "dst_path": dst_path,
                "error": "dst_path must be relative to the workspace with a directory component",
            }
        if Path(dst_path).is_absolute():
            return {"dst_path": dst_path, "error": "dst_path must be relative to the workspace"}
        workspace_dir = Path(self.file_store.workspace_path or ".").resolve()
        dst_abs = (workspace_dir / dst_path).resolve()
        try:
            dst_abs.relative_to(workspace_dir)
        except ValueError:
            return {"dst_path": dst_path, "error": "dst_path must stay inside the workspace"}
        if dst_abs == src_abs:
            return {"src_path": src_path, "dst_path": dst_path, "error": "src_path and dst_path are the same"}
        if dst_abs.is_dir():
            return {"dst_path": dst_path, "error": "destination is a directory"}
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
