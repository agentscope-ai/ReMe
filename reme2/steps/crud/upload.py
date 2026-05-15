"""``file_upload`` — copy a local file into the vault.

Agent flow: agent has a file at some local path (a download result, a
freshly generated artifact, a user-supplied attachment), and wants it
to live inside the vault at a known location. This is type-agnostic
— text, binary, anything — so it's the entry point for non-markdown
materials too.

The watcher / parser pick up the new file asynchronously; this step
just performs the copy. ``overwrite=True`` is the default since most
upload flows are intentional replacements (re-uploading the same
material after re-generation).
"""

from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path

from agentscope.tool import ToolResponse

from .download import resolve_path

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


@R.register("file_upload")
class FileUpload(BaseStep):
    """Copy a local file into the vault. Watcher / parser register the
    FileNode asynchronously."""

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
        src = Path(local_path)
        if not src.is_file():
            return {"local_path": local_path, "error": "source not found"}
        dst = resolve_path(self.file_store, path)
        if dst.exists() and not overwrite:
            return {"path": path, "error": "destination exists; pass overwrite=True"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return {
            "path": path,
            "size": dst.stat().st_size,
            "mime": mimetypes.guess_type(dst.name)[0] or "application/octet-stream",
        }
