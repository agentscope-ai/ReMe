"""``file_upload`` — copy a local file into the vault at a given path.

Used to put materials (or any binary/text artifact) into the vault
to live inside the vault at a known location. This is type-agnostic
— text, binary, anything — so it's the entry point for non-markdown
materials too.

The watcher / parser pick up the new file asynchronously; this step
just performs the copy. ``overwrite=True`` is the default since most
upload flows are intentional replacements (re-uploading the same
material after re-generation).

``path`` **must be a vault-relative path** with a directory component
— short links and absolute paths are rejected. File creation
needs an unambiguous primary key, and short links can't promise that
without a graph entry. Use ``file_move`` to rename existing files.
"""

from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum
from ...utils import path_resolver


@R.register("file_upload")
class FileUpload(BaseStep):
    """Copy ``local_path`` into the vault at ``path``."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        local_path: str = self.context.get("local_path", "") or ""
        path: str = self.context.get("path", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", True))
        assert local_path and path, "local_path and path are required"
        payload = self._upload(local_path, path, overwrite)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def file_upload(
        self, local_path: str, path: str, overwrite: bool = True,
    ) -> ToolResponse:
        """Copy ``local_path`` into the vault at ``path``."""
        payload = self._upload(local_path, path, overwrite)
        ok = "error" not in payload
        return _tool_response("file_upload", ok, payload, audit=self.audit)

    def _upload(self, local_path: str, path: str, overwrite: bool) -> dict:
        if Path(path).is_absolute():
            return {"path": path, "error": "path must be vault-relative"}
        if path_resolver.is_short_path(path):
            return {"path": path, "error": "path must include a directory component"}
        src = Path(local_path)
        if not src.is_file():
            return {"local_path": local_path, "error": "source not found"}
        dst = path_resolver.to_absolute(self.file_store, path)
        if dst.exists() and not overwrite:
            return {"path": path, "error": "destination exists; pass overwrite=True"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return {
            "path": path,
            "size": dst.stat().st_size,
            "mime": mimetypes.guess_type(dst.name)[0] or "application/octet-stream",
        }
