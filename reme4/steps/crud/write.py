"""Write a markdown file with a small, fixed front matter (``name``, ``description``)."""

import frontmatter

from ._file_io import NON_MD_WARNING, detect_file_encoding, gate_md, resolve_path, write_file_safe
from ..base_step import BaseStep
from ...components import R


@R.register("write_step")
class WriteStep(BaseStep):
    """Write (create or overwrite) a markdown file. When the target already exists,
    its contents are replaced and a system notice is appended to the answer.

    Front matter is restricted to two string fields: ``name`` and ``description``.
    The CLI schema declares them as required, but the step itself is lenient —
    missing or empty values are silently skipped so manual invocations don't
    fail catastrophically."""

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
        content = "" if content is None else str(content)

        target, err = resolve_path(self.working_path, raw)
        if err:
            self._fail(err)
            return None

        target, is_md = gate_md(target)

        existed = target.exists()

        # Non-markdown files have no frontmatter convention: name/description
        # are silently dropped and the body is written verbatim.
        if is_md:
            meta: dict = {}
            for key in ("name", "description"):
                value = self.context.get(key)
                if value is None:
                    continue
                s = str(value).strip()
                if not s:
                    continue
                meta[key] = s

            if meta:
                post = frontmatter.Post(content, **meta)
                body = frontmatter.dumps(post)
            else:
                body = content
            if not body.endswith("\n"):
                body += "\n"
        else:
            body = content

        # Preserve the existing file's encoding when overwriting (e.g. GBK CSV
        # stays GBK). New files are written as UTF-8.
        encoding = await detect_file_encoding(target) if existed else "utf-8"
        try:
            await write_file_safe(target, body, encoding=encoding)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"write failed: {e}", path=str(target))
            return None

        try:
            nbytes = len(body.encode(encoding))
        except (UnicodeEncodeError, LookupError):
            nbytes = len(body.encode("utf-8"))
        self.context.response.success = True
        if existed:
            answer = f"Wrote {target} ({nbytes} bytes) " f"[system notice: target already existed and was overwritten]"
        else:
            answer = f"Wrote {target} ({nbytes} bytes)"
        if not is_md:
            answer = f"{answer} [system notice: {NON_MD_WARNING}]"
        self.context.response.answer = answer
        self.logger.info(
            f"[{self.name}] wrote path={target} bytes={nbytes} encoding={encoding} "
            f"overwritten={existed} is_md={is_md}",
        )
        return self.context.response
