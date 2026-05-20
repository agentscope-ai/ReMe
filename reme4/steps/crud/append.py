"""Append content to the end of a markdown file (auto-creates if missing)."""

import aiofiles
import aiofiles.os

from ._file_io import gate_md, resolve_path
from ..base_step import BaseStep
from ...components import R


@R.register("append_step")
class AppendStep(BaseStep):
    """Append `content` to the target file. If the file does not exist it is
    created (a system notice is appended to the answer in that case).

    Uses native append mode (``ab``) so we don't have to load the file body
    just to add a few bytes. When the file already exists we peek its last
    byte to decide whether a leading newline is needed — guarding against
    fusing the new text into the previous line."""

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
        try:
            if created:
                target.parent.mkdir(parents=True, exist_ok=True)
                needs_newline = False
            else:
                size = (await aiofiles.os.stat(str(target))).st_size
                if size == 0:
                    needs_newline = False
                else:
                    async with aiofiles.open(str(target), "rb") as f:
                        await f.seek(size - 1)
                        needs_newline = (await f.read(1)) != b"\n"

            async with aiofiles.open(str(target), "ab") as f:
                if needs_newline:
                    await f.write(b"\n")
                await f.write(content_str.encode("utf-8"))
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
