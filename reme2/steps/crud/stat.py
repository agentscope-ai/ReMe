"""``file_stat`` — peek at vault file metadata without copying it.

Cheap inspection alternative to ``file_download``: the agent gets
size, mtime, mime type, and (for markdown files) the parsed
frontmatter — enough to decide whether to download / parse / skip
without paying the copy cost.

Returns a uniform envelope:

    {exists, type, size, mtime, ctime, mime, frontmatter}

``exists=False`` short-circuits everything else to ``None``. ``type``
is ``"file"`` / ``"dir"`` (covers event workspace probes too).
``frontmatter`` is populated only for ``.md`` files and only when
parsing succeeds — schema validity is a lint concern.
"""

from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path

import frontmatter
from agentscope.tool import ToolResponse

from .download import resolve_path

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


def _iso(ts: float) -> str:
    """Filesystem timestamp → ISO 8601 string."""
    return datetime.fromtimestamp(ts).isoformat()


def _stat(file_store, path: str) -> dict:
    target = resolve_path(file_store, path)
    if not target.exists():
        return {"path": path, "exists": False}

    st = target.stat()
    out: dict = {
        "path": path,
        "path": str(target),
        "exists": True,
        "type": "dir" if target.is_dir() else "file",
        "mtime": _iso(st.st_mtime),
        "ctime": _iso(st.st_ctime),
    }

    if target.is_file():
        out["size"] = st.st_size
        out["mime"] = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".md":
            try:
                meta = dict(frontmatter.loads(target.read_text(encoding="utf-8")).metadata)
            except Exception:
                meta = {}
            out["frontmatter"] = meta

    return out


@R.register("file_stat")
class FileStat(BaseStep):
    """Return metadata for a vault file or directory."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"
        payload = _stat(self.file_store, path)
        self.context.response.success = payload.get("exists", False)
        _set_answer(self.context, payload)

    async def file_stat(self, path: str) -> ToolResponse:
        """Return metadata for a vault file or directory.

        Returns ``{exists, type, size, mtime, ctime, mime, frontmatter}``.
        ``size`` / ``mime`` / ``frontmatter`` are file-only;
        ``frontmatter`` is markdown-only.
        """
        payload = _stat(self.file_store, path)
        ok = payload.get("exists", False)
        return _tool_response("file_stat", ok, payload, audit=self.audit)
