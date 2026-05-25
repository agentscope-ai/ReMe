"""``daily_write`` — write a daily note's full body + frontmatter; refresh day index.

Collapses the old ``daily_resolve`` → ``daily_create`` → ``file_write``
chain into a single call. Validates the slug, mkdirs the day folder,
writes body + frontmatter in one shot, refreshes ``daily/<date>.md``
index.

The ``overwrite`` flag picks between two behaviours:

* ``overwrite=False`` (default) — idempotent create: when the note
  already exists returns ``{created: False, overwritten: False}``
  without touching it (mirrors the old ``daily_resolve`` semantics).
  Index still refreshes (siblings may have changed; cheap self-healing).
* ``overwrite=True`` — unconditional write. Preserves existing-file
  encoding via ``detect_file_encoding`` (mirrors ``write_step``).

Frontmatter input is a dict; defaults to ``{name: slug}``. Empty /
None values are dropped (mirrors ``write_step``'s lenient frontmatter
handling). For partial frontmatter mutations on an existing note use
``frontmatter_update`` instead — ``daily_write`` is a full-file C/U.
"""

from datetime import date as _date

import frontmatter

from ._daily_io import refresh_day_index, validate_slug
from ..crud._file_io import detect_file_encoding, write_file_safe
from ..base_step import BaseStep
from ...components import R


@R.register("daily_write_step")
class DailyWriteStep(BaseStep):
    """Write ``daily/<date>/<slug>.md`` (create/overwrite); refresh day index."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    async def execute(self):  # pylint: disable=too-many-return-statements
        assert self.context is not None
        slug: str = self.context.get("slug", "") or ""
        body: str = self.context.get("body", "") or ""
        day: str = (self.context.get("date") or "").strip() or _date.today().isoformat()
        overwrite: bool = bool(self.context.get("overwrite", False))
        refresh_index: bool = bool(self.context.get("refresh_index", True))

        fm_input = self.context.get("frontmatter")
        if fm_input is None:
            meta_in: dict = {"name": slug}
        elif isinstance(fm_input, dict):
            meta_in = dict(fm_input)
            meta_in.setdefault("name", slug)
        else:
            self._fail(f"frontmatter must be a dict, got {type(fm_input).__name__}")
            return None

        err = validate_slug(slug)
        if err:
            self._fail(err)
            return None

        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        path_rel = f"{daily_dir}/{day}/{slug}.md"
        path_abs = (self.vault_path / path_rel).resolve()
        existed = path_abs.is_file()

        # overwrite=False + file exists → idempotent skip (matches old daily_resolve semantics).
        if existed and not overwrite:
            payload: dict = {
                "date": day,
                "slug": slug,
                "path": path_rel,
                "created": False,
                "overwritten": False,
            }
            if refresh_index:
                payload["index"] = await refresh_day_index(self.file_store, day, daily_dir)
            self.context.response.success = True
            self.context.response.answer = f"Reused existing daily note {path_rel}"
            self.context.response.metadata.update(payload)
            return self.context.response

        # Build the post. Drop empty / None values (write_step idiom).
        clean_meta: dict = {}
        for k, v in meta_in.items():
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            clean_meta[k] = v
        post = frontmatter.Post(body, **clean_meta)
        text = frontmatter.dumps(post)
        if not text.endswith("\n"):
            text += "\n"

        # Preserve existing file encoding on overwrite; new files = UTF-8.
        encoding = await detect_file_encoding(path_abs) if existed else "utf-8"
        try:
            await write_file_safe(path_abs, text, encoding=encoding)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"write failed: {e}", date=day, slug=slug, path=path_rel)
            return None

        payload = {
            "date": day,
            "slug": slug,
            "path": path_rel,
            "created": not existed,
            "overwritten": existed,
        }
        if refresh_index:
            payload["index"] = await refresh_day_index(self.file_store, day, daily_dir)

        self.context.response.success = True
        verb = "Wrote" if not existed else "Overwrote"
        self.context.response.answer = f"{verb} daily note {path_rel}"
        self.context.response.metadata.update(payload)
        try:
            nbytes = len(text.encode(encoding))
        except (UnicodeEncodeError, LookupError):
            nbytes = len(text.encode("utf-8"))
        self.logger.info(
            f"[{self.name}] wrote path={path_rel} bytes={nbytes} "
            f"overwrite={overwrite} existed={existed} refresh_index={refresh_index}",
        )
        return self.context.response
