"""Find-and-replace text in a markdown file body (front matter is preserved)."""

import frontmatter
import yaml

from ._file_io import get_path_lock, read_file_safe, write_file_safe
from ._path import NON_MD_WARNING, gate_md, resolve_path
from ..base_step import BaseStep
from ...components import R


@R.register("edit_step")
class EditStep(BaseStep):
    """Replace every occurrence of ``old`` with ``new`` inside the file body.

    The YAML front matter block (if any) is parsed out, kept verbatim and
    re-emitted unchanged — matches that fall inside front matter are ignored,
    so a typo in `old` cannot corrupt structured metadata.

    Concurrency: in-process per-path ``asyncio.Lock`` serializes the
    read-modify-write cycle against the same file (multi-worker / multi-
    process safety is out of scope)."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    async def execute(self):  # pylint: disable=too-many-return-statements
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        old = self.context.get("old")
        new = self.context.get("new")

        if old is None or str(old) == "":
            self._fail("`old` is required and must be non-empty")
            return None
        if new is None:
            self._fail("`new` is required")
            return None
        old_str = str(old)
        new_str = str(new)

        target, err = resolve_path(self.vault_path, raw)
        if err:
            self._fail(err)
            return None

        target, is_md = gate_md(target)

        lock = await get_path_lock(target)
        async with lock:
            if not target.exists():
                self._fail(f"file {target} does not exist", path=str(target))
                return None
            if not target.is_file():
                self._fail(f"path {target} is not a file", path=str(target))
                return None

            try:
                raw_text, encoding = await read_file_safe(target)
            except Exception as e:  # pylint: disable=broad-except
                self._fail(f"read failed: {e}", path=str(target))
                return None

            # Markdown: parse frontmatter and operate on body only. Non-markdown:
            # there's no frontmatter convention, so operate on the full text.
            if is_md:
                try:
                    post = frontmatter.loads(raw_text)
                except yaml.YAMLError as exc:
                    self._fail(f"failed to parse frontmatter in {target}: {exc}", path=str(target))
                    return None
                body = post.content
                not_found_msg = (
                    f"text to replace was not found in the body of {target} (front matter is excluded from edit)"
                )
            else:
                post = None
                body = raw_text
                not_found_msg = f"text to replace was not found in {target}"

            if old_str not in body:
                self._fail(not_found_msg, path=str(target))
                return None

            count = body.count(old_str)
            new_body = body.replace(old_str, new_str)

            if is_md and post is not None:
                post.content = new_body
                # Re-serialize: keep front matter when present, otherwise emit body alone
                # so we don't introduce an empty `---\n---\n` block.
                new_text = frontmatter.dumps(post) if post.metadata else post.content
                if not new_text.endswith("\n"):
                    new_text += "\n"
            else:
                new_text = new_body

            # Preserve the file's original encoding (returned by read_file_safe above)
            # so edits don't silently re-encode non-UTF-8 files (e.g. GBK CSV) to UTF-8.
            try:
                await write_file_safe(target, new_text, encoding=encoding)
            except Exception as e:  # pylint: disable=broad-except
                self._fail(f"write failed: {e}", path=str(target))
                return None

        self.context.response.success = True
        answer = f"Replaced {count} occurrence(s) in {target}"
        if not is_md:
            answer = f"{answer} [system notice: {NON_MD_WARNING}]"
        self.context.response.answer = answer
        self.logger.info(f"[{self.name}] edited path={target} count={count} is_md={is_md}")
        return self.context.response
