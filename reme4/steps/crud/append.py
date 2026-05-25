"""Append content to the end of a file (auto-creates if missing)."""

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


@R.register("append_step")
class AppendStep(BaseStep):
    """Append `content` to the target file. If the file does not exist it is
    created (a system notice is appended to the answer in that case).

    Content is appended verbatim — callers control whether a separator newline
    is included in the input.

    Concurrency: append is implemented as read-concat-atomic_replace through
    OCC, so concurrent writers cannot interleave bytes mid-line. Greenfield
    creates skip OCC recheck (last writer wins for the create race only);
    appends to existing files are retried (up to 10x with backoff) when the
    file is modified between read and replace.
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
        content_str = "" if content is None else str(content)

        target, err = resolve_path(self.working_path, raw)
        if err:
            self._fail(err)
            return None

        target, is_md = gate_md(target)

        if target.exists() and not target.is_file():
            self._fail(f"path {target} is not a file", path=str(target))
            return None

        existed_at_start = target.exists()
        state = {"appended_bytes": 0, "encoding": "utf-8"}

        async def compute(old_bytes, _v0):
            if old_bytes is None:
                # Greenfield: encode the content alone as utf-8.
                payload = content_str.encode("utf-8")
                state["appended_bytes"] = len(payload)
                state["encoding"] = "utf-8"
                return payload
            # Preserve the file's existing encoding so appended bytes don't
            # corrupt a non-UTF-8 file (e.g. GBK CSV).
            text, enc = decode_known_file(old_bytes, target.suffix)
            new_text = text + content_str
            try:
                appended = content_str.encode(enc)
                new_payload = new_text.encode(enc)
            except (UnicodeEncodeError, LookupError):
                enc = "utf-8"
                appended = content_str.encode(enc)
                new_payload = new_text.encode(enc)
            state["appended_bytes"] = len(appended)
            state["encoding"] = enc
            return new_payload

        try:
            _nbytes, _v_final, attempts = await occ_write(target, compute, occ_on_create=False)
        except ConflictError as e:
            self._fail(
                f"append conflict on {target}: file was modified concurrently after {e.attempts} attempts",
                path=str(target),
                conflict=True,
                attempts=e.attempts,
            )
            return None
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"append failed: {e}", path=str(target))
            return None

        nbytes = state["appended_bytes"]
        encoding = state["encoding"]
        self.context.response.success = True
        if existed_at_start:
            answer = f"Appended {nbytes} bytes to {target}"
        else:
            answer = f"Appended {nbytes} bytes to {target} [system notice: file did not exist and was auto-created]"
        if not is_md:
            answer = f"{answer} [system notice: {NON_MD_WARNING}]"
        self.context.response.answer = answer
        self.context.response.metadata.update({"path": str(target), "attempts": attempts})
        self.logger.info(
            f"[{self.name}] appended path={target} bytes={nbytes} encoding={encoding} "
            f"created={not existed_at_start} is_md={is_md} attempts={attempts}",
        )
        return self.context.response
