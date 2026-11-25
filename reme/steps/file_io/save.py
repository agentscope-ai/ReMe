"""Save a text file verbatim with optional optimistic-concurrency protection."""

from contextlib import suppress
from datetime import datetime
from stat import S_ISREG

from ._base import _FileIOStep
from ._file_io import detect_file_encoding, write_file_safe
from ...utils.files.file_version import FileChangedError
from ...utils.files.path_lock import get_path_lock
from ._path import check_path_permission, resolve_path
from ...core.registry import R


@R.register("save_step")
class SaveStep(_FileIOStep):
    """Persist editor content without interpreting or rebuilding frontmatter.

    ``expected_mtime`` is the ISO timestamp returned by ``stat``. When supplied,
    saving fails if the file changed after it was opened. This keeps the
    filesystem authoritative when ReMe and another editor are used together.
    """

    async def execute(self):
        assert self.context is not None
        raw_path = str(self.context.get("path") or "")
        content = self.context.get("content")
        expected_mtime = self.context.get("expected_mtime")
        target, error = resolve_path(self.workspace_path, raw_path)
        if self.kwargs.get("require_expected_mtime") and (
            not isinstance(expected_mtime, str) or not expected_mtime or not isinstance(content, str)
        ):
            error = "content and expected_mtime are required to save a document"
        if error or target is None:
            self._fail(error or "invalid path", path=raw_path)
            return self.context.response
        if not check_path_permission(self.workspace_path, target, self.context.get("_allowed_paths")):
            self._fail("no permission to write this file", path=raw_path)
            return self.context.response
        lock = await get_path_lock(target)
        async with lock:
            original = None
            with suppress(FileNotFoundError):
                original = target.lstat()
            if original is not None and not S_ISREG(original.st_mode):
                self._fail("target is not a file", path=raw_path)
                return self.context.response
            current_mtime = datetime.fromtimestamp(original.st_mtime).isoformat() if original is not None else None
            try:
                if expected_mtime and current_mtime != str(expected_mtime):
                    raise FileChangedError(raw_path)
                encoding = await detect_file_encoding(target) if original is not None else "utf-8"
                await write_file_safe(
                    target,
                    str(content or ""),
                    encoding=encoding,
                    max_bytes=self.kwargs.get("max_bytes"),
                    expected_version=original if expected_mtime else None,
                )
            except FileChangedError:
                current_mtime = None
                with suppress(FileNotFoundError):
                    current_mtime = datetime.fromtimestamp(target.lstat().st_mtime).isoformat()
                self._fail(
                    "File changed outside ReMe. Reload it before saving.",
                    code="file_conflict",
                    path=raw_path,
                    expected_mtime=str(expected_mtime),
                    current_mtime=current_mtime,
                )
                return self.context.response
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
