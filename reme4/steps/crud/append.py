"""Append content to the end of a markdown file (auto-creates if missing)."""

from ._file_io import gate_md, read_file_safe, resolve_path, write_file_safe
from ..base_step import BaseStep
from ...components import R


@R.register("append_step")
class AppendStep(BaseStep):
    """Append `content` to the target file. If the file does not exist it is
    created (a system notice is appended to the answer in that case)."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    async def execute(self):  # pylint: disable=too-many-return-statements
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        content = self.context.get("content")
        content_str = "" if content is None else str(content)

        target, err = resolve_path(self.working_path, raw)
        if err:
            self._fail(err)
            return None

        target, err = gate_md(target, raw)
        if err:
            self._fail(err)
            return None

        if target.exists() and not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return None

        created = not target.exists()
        if created:
            new_content = content_str
        else:
            try:
                existing = await read_file_safe(target)
            except Exception as e:  # pylint: disable=broad-except
                self._fail(f"read failed: {e}", path=str(target))
                return None
            # Ensure a newline boundary so we don't fuse the appended text into the previous line.
            sep = "" if (not existing or existing.endswith("\n")) else "\n"
            new_content = existing + sep + content_str

        try:
            await write_file_safe(target, new_content)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"write failed: {e}", path=str(target))
            return None

        nbytes = len(content_str.encode("utf-8"))
        self.context.response.success = True
        if created:
            self.context.response.answer = (
                f"Appended {nbytes} bytes to {target} " f"[system notice: file did not exist and was auto-created]"
            )
        else:
            self.context.response.answer = f"Appended {nbytes} bytes to {target}"
        self.logger.info(
            f"[{self.name}] appended path={target} bytes={nbytes} created={created}",
        )
        return self.context.response
