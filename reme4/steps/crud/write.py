"""Write a markdown file with a small, fixed front matter (``name``, ``description``)."""

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


@R.register("write_step")
class WriteStep(BaseStep):
    """Write (create or overwrite) a markdown file.

    Frontmatter accepts two reserved string fields (``name`` / ``description``)
    plus a free-form ``metadata`` dict whose entries are expanded into the
    frontmatter as-is. On key collision, the explicit ``name`` / ``description``
    parameters take precedence over the same keys inside ``metadata``.

    Concurrency: writes go through OCC + atomic replace. When the file already
    exists at the start of the call, a (mtime_ns, size) stamp is taken and
    re-checked just before replace; on mismatch the loop retries (up to 10x
    with exponential backoff). Greenfield creates skip the recheck — last
    writer wins for the create race only.
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

        target, err = resolve_path(self.working_path, raw)
        if err:
            self._fail(err)
            return None

        target, is_md = gate_md(target)

        # Body is independent of file state — compose once outside the OCC loop.
        if is_md:
            meta: dict = {}
            # expand `metadata` dict (skip name/description; explicit params win).
            if isinstance(metadata_raw, dict):
                for k, v in metadata_raw.items():
                    if k in ("name", "description"):
                        continue
                    if v is None:
                        continue
                    meta[str(k)] = v
            # Pass 2: layer explicit name/description on top.
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

        existed_at_start = target.exists()
        encoding_used = {"enc": "utf-8"}

        async def compute(old_bytes, _v0):
            # Preserve the file's original encoding when overwriting; default
            # UTF-8 for greenfield creates. Re-derived each attempt so a
            # concurrent rewrite (e.g. utf-8 → gbk) is honored on retry.
            if old_bytes is not None:
                _, enc = decode_known_file(old_bytes, target.suffix)
            else:
                enc = "utf-8"
            try:
                payload = body.encode(enc)
            except (UnicodeEncodeError, LookupError):
                enc = "utf-8"
                payload = body.encode(enc)
            encoding_used["enc"] = enc
            return payload

        try:
            nbytes, _v_final, attempts = await occ_write(target, compute, occ_on_create=False)
        except ConflictError as e:
            self._fail(
                f"write conflict on {target}: file was modified concurrently after {e.attempts} attempts",
                path=str(target),
                conflict=True,
                attempts=e.attempts,
            )
            return None
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"write failed: {e}", path=str(target))
            return None

        encoding = encoding_used["enc"]
        self.context.response.success = True
        if existed_at_start:
            answer = f"Wrote {target} ({nbytes} bytes) [system notice: target already existed and was overwritten]"
        else:
            answer = f"Wrote {target} ({nbytes} bytes)"
        if not is_md:
            answer = f"{answer} [system notice: {NON_MD_WARNING}]"
        self.context.response.answer = answer
        self.context.response.metadata.update({"path": str(target), "attempts": attempts})
        self.logger.info(
            f"[{self.name}] wrote path={target} bytes={nbytes} encoding={encoding} "
            f"overwritten={existed_at_start} is_md={is_md} attempts={attempts}",
        )
        return self.context.response
