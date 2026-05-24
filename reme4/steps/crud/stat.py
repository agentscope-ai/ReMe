"""``file_stat`` — peek at file metadata under the vault without copying it.

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

``path`` accepts a file path or directory path relative to the vault.
Joined with ``file_store.vault_path`` and inspected on disk.
"""

from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path

import frontmatter

from ..base_step import BaseStep
from ...utils import set_answer

from ...components import R
from ...enumeration import ComponentEnum


def _iso(ts: float) -> str:
    """Filesystem timestamp → ISO 8601 string."""
    return datetime.fromtimestamp(ts).isoformat()


async def _stat(file_store, path: str) -> dict:
    target = (Path(file_store.vault_path or ".") / path).resolve()
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


@R.register("stat_step")
class StatStep(BaseStep):
    """Return metadata for a file or directory under the vault."""

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"
        payload = await _stat(self.file_store, path)
        self.context.response.success = payload.get("exists", False) and "error" not in payload
        set_answer(self.context, payload)
