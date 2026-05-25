"""Find-and-replace text in a markdown file body (front matter is preserved)."""

import frontmatter

from ._file_io import (
    NON_MD_WARNING,
    ConflictError,
    decode_known_file,
    gate_md,
    occ_write,
    resolve_path,
)
from ..base_step import BaseStep
from ...components import R


class _EditNotFoundError(Exception):
    """``old`` not present in file body — a hard error, not an OCC conflict."""


class _EditFileMissingError(Exception):
    """File disappeared while editing (caught inside the OCC loop)."""


@R.register("edit_step")
class EditStep(BaseStep):
    """Replace every occurrence of ``old`` with ``new`` inside the file body.

    The YAML front matter block (if any) is parsed out, kept verbatim and
    re-emitted unchanged — matches that fall inside front matter are ignored,
    so a typo in `old` cannot corrupt structured metadata.

    Concurrency: full read-modify-write goes through OCC + atomic replace.
    If ``old`` is not found in the current body, the operation fails
    immediately (no retry — re-reading won't help).
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

        target, err = resolve_path(self.working_path, raw)
        if err:
            self._fail(err)
            return None

        target, is_md = gate_md(target)

        if not target.exists():
            self._fail(f"file {target} does not exist", path=str(target))
            return None
        if not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return None

        count_holder = {"count": 0}

        async def compute(old_bytes, _v0):
            if old_bytes is None:
                # File was concurrently deleted between our existence check and
                # this read. Treat as a hard error, not an OCC retry.
                raise _EditFileMissingError()

            text, enc = decode_known_file(old_bytes, target.suffix)

            if is_md:
                post = frontmatter.loads(text)
                body = post.content
                not_found_msg = (
                    f"text to replace was not found in the body of {target} (front matter is excluded from edit)"
                )
            else:
                post = None
                body = text
                not_found_msg = f"text to replace was not found in {target}"

            if old_str not in body:
                raise _EditNotFoundError(not_found_msg)

            count_holder["count"] = body.count(old_str)
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

            try:
                return new_text.encode(enc)
            except (UnicodeEncodeError, LookupError):
                return new_text.encode("utf-8")

        try:
            nbytes, _v_final, attempts = await occ_write(target, compute, occ_on_create=True)
        except _EditNotFoundError as e:
            self._fail(str(e), path=str(target))
            return None
        except _EditFileMissingError:
            self._fail(f"file {target} no longer exists", path=str(target))
            return None
        except ConflictError as e:
            self._fail(
                f"edit conflict on {target}: file was modified concurrently after {e.attempts} attempts",
                path=str(target),
                conflict=True,
                attempts=e.attempts,
            )
            return None
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"edit failed: {e}", path=str(target))
            return None

        count = count_holder["count"]
        self.context.response.success = True
        answer = f"Replaced {count} occurrence(s) in {target}"
        if not is_md:
            answer = f"{answer} [system notice: {NON_MD_WARNING}]"
        self.context.response.answer = answer
        self.context.response.metadata.update({"path": str(target), "attempts": attempts, "bytes": nbytes})
        self.logger.info(
            f"[{self.name}] edited path={target} count={count} is_md={is_md} attempts={attempts} bytes={nbytes}",
        )
        return self.context.response
