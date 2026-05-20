"""Write a markdown file with a small, fixed front matter (``name``, ``description``)."""

import frontmatter

from ._file_io import gate_md, resolve_path, write_file_safe
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

        target, err = gate_md(target, raw)
        if err:
            self._fail(err)
            return None

        existed = target.exists()

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

        try:
            await write_file_safe(target, body)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"write failed: {e}", path=str(target))
            return None

        nbytes = len(body.encode("utf-8"))
        self.context.response.success = True
        if existed:
            self.context.response.answer = (
                f"Wrote {target} ({nbytes} bytes) " f"[system notice: target already existed and was overwritten]"
            )
        else:
            self.context.response.answer = f"Wrote {target} ({nbytes} bytes)"
        self.logger.info(
            f"[{self.name}] wrote path={target} bytes={nbytes} overwritten={existed}",
        )
        return self.context.response
