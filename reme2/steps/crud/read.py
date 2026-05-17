"""``read`` — return the text content of a vault file.

Counterpart to ``write``. Suitable for any operation that needs the
full file body (markdown or otherwise) — wikilink edits, workspace
read-modify-write loops, etc.

``path`` accepts a short link or vault-relative path; resolution
goes through ``path_resolver.resolve_to_absolute``. Short-link
ambiguity is surfaced as ``error="ambiguous"`` with the candidate
list — the read is **not** executed in that case.

Returns ``{exists, content}`` (``content`` omitted when ``exists=False``).
"""

from __future__ import annotations

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum
from ...utils import path_resolver


async def _read(file_store, path: str, encoding: str) -> dict:
    try:
        target = await path_resolver.resolve_to_absolute(file_store, path)
    except path_resolver.PathAmbiguous as e:
        return {"path": path, "error": "ambiguous", "candidates": e.candidates}
    except path_resolver.PathNotFound:
        return {"path": path, "exists": False}
    if not target.is_file():
        return {"path": path, "exists": False}
    return {
        "path": path,
        "exists": True,
        "content": target.read_text(encoding=encoding),
    }


@R.register("read")
class FileRead(BaseStep):
    """Return the text content of a vault file."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        encoding: str = self.context.get("encoding") or "utf-8"
        assert path, "path is required"
        payload = await _read(self.file_store, path, encoding)
        self.context.response.success = payload.get("exists", False) and "error" not in payload
        _set_answer(self.context, payload)

    async def file_read(self, path: str, encoding: str = "utf-8") -> ToolResponse:
        """Return the text content of the vault file at ``path``."""
        payload = await _read(self.file_store, path, encoding)
        ok = payload.get("exists", False) and "error" not in payload
        return _tool_response("read", ok, payload, audit=self.audit)
