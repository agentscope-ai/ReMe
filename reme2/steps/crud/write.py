"""``write`` — write text content to a vault file.

Counterpart to ``read``. Whole-file replace (or create); use this for
any operation that needs to put a string of bytes at a known path.
The watcher / parser pick up the change asynchronously.

``path`` **must be a vault-relative path** with a directory component
— short links and absolute paths are rejected (same rule as
``upload``). File creation needs an unambiguous primary key, and a
short link can't promise that without a graph entry.

``overwrite=True`` is the default since most write flows are
intentional replacements.
"""

from __future__ import annotations

from pathlib import Path

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum
from ...utils import path_resolver


def _write(file_store, path: str, content: str, overwrite: bool, encoding: str) -> dict:
    if Path(path).is_absolute():
        return {"path": path, "error": "path must be vault-relative"}
    if path_resolver.is_short_path(path):
        return {"path": path, "error": "path must include a directory component"}
    target = path_resolver.to_absolute(file_store, path)
    if target.exists() and not overwrite:
        return {"path": path, "error": "destination exists; pass overwrite=True"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding=encoding)
    return {"path": path, "size": target.stat().st_size}


@R.register("write")
class FileWrite(BaseStep):
    """Write text content to a vault file."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        content: str = self.context.get("content", "") or ""
        overwrite: bool = bool(self.context.get("overwrite", True))
        encoding: str = self.context.get("encoding") or "utf-8"
        assert path, "path is required"
        payload = _write(self.file_store, path, content, overwrite, encoding)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def file_write(
        self,
        path: str,
        content: str,
        overwrite: bool = True,
        encoding: str = "utf-8",
    ) -> ToolResponse:
        """Write ``content`` to the vault file at ``path``."""
        payload = _write(self.file_store, path, content, overwrite, encoding)
        ok = "error" not in payload
        return _tool_response("write", ok, payload, audit=self.audit)
