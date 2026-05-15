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

``path`` accepts a short link, vault-relative path, or directory
path. File-style inputs go through ``path_resolver.resolve_to_absolute``
so short-link ambiguity surfaces as ``error="ambiguous"`` (with
candidates) and the call is **not** executed. Directory paths and
non-indexed files fall back to a plain ``working_dir`` join.
"""

from __future__ import annotations

import mimetypes
from datetime import datetime

import frontmatter
from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum
from ...utils import path_resolver


def _iso(ts: float) -> str:
    """Filesystem timestamp → ISO 8601 string."""
    return datetime.fromtimestamp(ts).isoformat()


async def _stat(file_store, path: str) -> dict:
    try:
        target = await path_resolver.resolve_to_absolute(file_store, path)
    except path_resolver.PathAmbiguous as e:
        return {"path": path, "error": "ambiguous", "candidates": e.candidates}
    except path_resolver.PathNotFound:
        # Directory or unindexed path — fall back to plain join.
        target = path_resolver.to_absolute(file_store, path)

    if not target.exists():
        return {"path": path, "exists": False}

    st = target.stat()
    out: dict = {
        "path": path,
        "absolute_path": str(target),
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


@R.register("stat")
class FileStat(BaseStep):
    """Return metadata for a vault file or directory."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"
        payload = await _stat(self.file_store, path)
        self.context.response.success = payload.get("exists", False) and "error" not in payload
        _set_answer(self.context, payload)

    async def file_stat(self, path: str) -> ToolResponse:
        """Return metadata for a vault file or directory.

        Returns ``{exists, type, size, mtime, ctime, mime, frontmatter}``.
        ``size`` / ``mime`` / ``frontmatter`` are file-only;
        ``frontmatter`` is markdown-only.
        """
        payload = await _stat(self.file_store, path)
        ok = payload.get("exists", False) and "error" not in payload
        return _tool_response("file_stat", ok, payload, audit=self.audit)
