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

import mimetypes
from datetime import datetime
from pathlib import Path

import frontmatter

from ..base_step import BaseStep
from ...components import R


@R.register("stat_step")
class StatStep(BaseStep):
    """Return metadata for a file or directory under the vault."""

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path", "") or ""
        assert path, "path is required"

        target = (Path(self.file_store.vault_path or ".") / path).resolve()
        if not target.exists():
            self.context.response.success = False
            self.context.response.answer = f"stat: {path} not found"
            self.context.response.metadata.update({"path": path, "exists": False})
            self.logger.info(f"[{self.name}] path={path} exists=False")
            return

        st = target.stat()
        payload: dict = {
            "path": path,
            "absolute_path": str(target),
            "exists": True,
            "type": "dir" if target.is_dir() else "file",
            "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "ctime": datetime.fromtimestamp(st.st_ctime).isoformat(),
        }
        if target.is_file():
            payload["size"] = st.st_size
            payload["mime"] = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".md":
                try:
                    meta = dict(frontmatter.loads(target.read_text(encoding="utf-8")).metadata)
                except Exception:
                    meta = {}
                payload["frontmatter"] = meta
            answer = f"stat: {path} (file, {st.st_size} bytes)"
        else:
            answer = f"stat: {path} (dir)"

        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(payload)
        self.logger.info(
            f"[{self.name}] path={path} type={payload['type']} "
            f"size={payload.get('size', '-')} mime={payload.get('mime', '-')}",
        )
