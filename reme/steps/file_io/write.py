"""Write a markdown file with a small, fixed front matter (``name``, ``description``)."""

import frontmatter

from ._file_io import detect_file_encoding, get_path_lock, write_file_safe
from ._path import NON_MD_WARNING, gate_md, resolve_path
from ..base_step import BaseStep
from ...components import R


@R.register("write_step")
class WriteStep(BaseStep):
    """Write (create or overwrite) a markdown file.

    Frontmatter accepts two reserved string fields (``name`` / ``description``)
    plus an optional free-form ``metadata`` dict whose entries are expanded
    into the frontmatter as-is. ``name`` / ``description`` keys inside
    ``metadata`` are ignored — only the top-level explicit parameters are
    honored for those two reserved fields.

    Concurrency: in-process per-path ``asyncio.Lock`` serializes concurrent
    writes to the same file (multi-worker / multi-process safety is out of
    scope).
    """

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
        metadata_raw = self.context.get("metadata")

        target, err = resolve_path(self.vault_path, raw)
        if err:
            self._fail(err)
            return None

        target, is_md = gate_md(target)

        # Non-markdown files have no frontmatter convention: name/description
        # and metadata are silently dropped and the body is written verbatim.
        if is_md:
            meta: dict = {}
            # Expand `metadata` dict first, skipping the two reserved keys.
            if isinstance(metadata_raw, dict):
                for k, v in metadata_raw.items():
                    if k in ("name", "description"):
                        continue
                    if v is None:
                        continue
                    meta[str(k)] = v
            # Layer explicit name/description on top (always wins).
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

        lock = await get_path_lock(target)
        async with lock:
            existed = target.exists()
            # Preserve the existing file's encoding when overwriting (e.g. GBK
            # CSV stays GBK). New files are written as UTF-8.
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
            answer = f"Wrote {target} ({nbytes} bytes) [system notice: target already existed and was overwritten]"
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
