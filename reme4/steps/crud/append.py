"""Append content to the end of a file (auto-creates if missing)."""

import aiofiles

from ._file_io import NON_MD_WARNING, detect_file_encoding, gate_md, resolve_path
from ..base_step import BaseStep
from ...components import R


@R.register("append_step")
class AppendStep(BaseStep):
    """Append `content` to the target file. If the file does not exist it is
    created (a system notice is appended to the answer in that case).

    Content is appended verbatim — callers control whether a separator newline
    is included in the input."""

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

        target, is_md = gate_md(target)

        if target.exists() and not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return None

        created = not target.exists()
        # Preserve the existing file's encoding so appended bytes don't corrupt
        # a non-UTF-8 file (e.g. GBK CSV). New files default to UTF-8.
        encoding = "utf-8" if created else await detect_file_encoding(target)
        try:
            if created:
                target.parent.mkdir(parents=True, exist_ok=True)
            try:
                payload = content_str.encode(encoding)
            except (UnicodeEncodeError, LookupError):
                self.logger.warning(
                    f"[{self.name}] cannot encode appended content as {encoding!r}, falling back to utf-8",
                )
                payload = content_str.encode("utf-8")
            async with aiofiles.open(str(target), "ab") as f:
                await f.write(payload)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"write failed: {e}", path=str(target))
            return None

        nbytes = len(payload)
        self.context.response.success = True
        if created:
            answer = f"Appended {nbytes} bytes to {target} [system notice: file did not exist and was auto-created]"
        else:
            answer = f"Appended {nbytes} bytes to {target}"
        if not is_md:
            answer = f"{answer} [system notice: {NON_MD_WARNING}]"
        self.context.response.answer = answer
        self.logger.info(
            f"[{self.name}] appended path={target} bytes={nbytes} encoding={encoding} "
            f"created={created} is_md={is_md}",
        )
        return self.context.response
