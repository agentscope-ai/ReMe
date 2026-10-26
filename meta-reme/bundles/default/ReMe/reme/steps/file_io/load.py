"""Load a complete text file for an interactive editor."""

from datetime import datetime

from ._file_io import read_file_safe
from ._path import _check_path_permission, resolve_path
from ..base_step import BaseStep
from ...components import R

DEFAULT_EDITOR_MAX_BYTES = 5 * 1024 * 1024


@R.register("load_step")
class LoadStep(BaseStep):
    """Return complete text without the truncation used by agent-facing reads."""

    def _fail(self, message: str, **metadata) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        self.context.response.metadata.update(metadata)

    async def execute(self):
        assert self.context is not None
        raw_path = str(self.context.get("path") or "")
        max_bytes = int(self.context.get("max_bytes") or self.kwargs.get("max_bytes") or DEFAULT_EDITOR_MAX_BYTES)
        target, error = resolve_path(self.workspace_path, raw_path)
        if error or target is None:
            self._fail(error or "invalid path", path=raw_path)
            return self.context.response
        if not _check_path_permission(self.workspace_path, target, self.context.get("_allowed_paths")):
            self._fail("no permission to read this file", path=raw_path)
            return self.context.response
        if not target.is_file():
            self._fail("file does not exist", path=raw_path)
            return self.context.response

        stat = target.stat()
        if stat.st_size > max_bytes:
            self._fail(
                f"file is too large for the editor ({stat.st_size} bytes; limit {max_bytes})",
                code="file_too_large",
                path=raw_path,
                size=stat.st_size,
                max_bytes=max_bytes,
            )
            return self.context.response
        try:
            content, encoding = await read_file_safe(target, max_bytes=max_bytes)
        except Exception as exc:  # pylint: disable=broad-except
            self._fail(f"load failed: {exc}", path=raw_path)
            return self.context.response

        self.context.response.success = True
        self.context.response.answer = content
        self.context.response.metadata.update(
            {
                "path": raw_path,
                "exists": True,
                "type": "file",
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "encoding": encoding,
            },
        )
        return self.context.response
