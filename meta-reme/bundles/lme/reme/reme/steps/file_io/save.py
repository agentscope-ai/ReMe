"""Save a text file verbatim with optional optimistic-concurrency protection."""

from datetime import datetime

from ._file_io import detect_file_encoding, get_path_lock, write_file_safe
from ._path import _check_path_permission, resolve_path
from ..base_step import BaseStep
from ...components import R


@R.register("save_step")
class SaveStep(BaseStep):
    """Persist editor content without interpreting or rebuilding frontmatter.

    ``expected_mtime`` is the ISO timestamp returned by ``stat``. When supplied,
    saving fails if the file changed after it was opened. This keeps the
    filesystem authoritative when ReMe and another editor are used together.
    """

    def _fail(self, message: str, **metadata) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        self.context.response.metadata.update(metadata)

    async def execute(self):
        assert self.context is not None
        raw_path = str(self.context.get("path") or "")
        content = str(self.context.get("content") or "")
        expected_mtime = self.context.get("expected_mtime")

        target, error = resolve_path(self.workspace_path, raw_path)
        if error or target is None:
            self._fail(error or "invalid path", path=raw_path)
            return self.context.response
        if not _check_path_permission(self.workspace_path, target, self.context.get("_allowed_paths")):
            self._fail("no permission to write this file", path=raw_path)
            return self.context.response
        if target.exists() and not target.is_file():
            self._fail("target is not a file", path=raw_path)
            return self.context.response

        lock = await get_path_lock(target)
        async with lock:
            current_mtime = datetime.fromtimestamp(target.stat().st_mtime).isoformat() if target.exists() else None
            if expected_mtime and current_mtime != str(expected_mtime):
                self._fail(
                    "file changed outside ReMe; reload it before saving",
                    code="file_conflict",
                    path=raw_path,
                    expected_mtime=str(expected_mtime),
                    current_mtime=current_mtime,
                )
                return self.context.response

            encoding = await detect_file_encoding(target) if target.exists() else "utf-8"
            try:
                await write_file_safe(target, content, encoding=encoding)
            except Exception as exc:  # pylint: disable=broad-except
                self._fail(f"save failed: {exc}", path=raw_path)
                return self.context.response

            stat = target.stat()
            saved_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

        self.context.response.success = True
        self.context.response.answer = f"Saved {raw_path} ({stat.st_size} bytes)"
        self.context.response.metadata.update(
            {
                "path": raw_path,
                "exists": True,
                "type": "file",
                "size": stat.st_size,
                "mtime": saved_mtime,
            },
        )
        self.logger.info(f"[{self.name}] saved path={target} bytes={stat.st_size} encoding={encoding}")
        return self.context.response
