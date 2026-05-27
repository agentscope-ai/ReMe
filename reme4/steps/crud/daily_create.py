"""``daily_create`` — provision an empty ``daily/<date>/<slug>.md``; return its path.

Minimal slug provisioner. Validates the slug, mkdirs the day folder,
writes an empty-body note with frontmatter ``{name: slug}`` if (and
only if) the file does not already exist, refreshes the day index,
and returns the vault-relative path.

Idempotent: when the note already exists this is a no-op write (the
day index still refreshes — siblings may have changed; cheap
self-healing). The caller fills the body via ``file_write`` /
``file_edit`` / ``file_append`` (or a native editor); ``daily_create``
deliberately does not accept a body.

Inputs:
    slug (required, validated)
    date (default today, ISO ``YYYY-MM-DD``)

Outputs:
    answer   = one-line human-readable status
    metadata = {date, slug, path, created, index?}
"""

from datetime import date as _date

import frontmatter

from ._file_io import refresh_day_index, validate_slug, write_file_safe
from ..base_step import BaseStep
from ...components import R


@R.register("daily_create_step")
class DailyCreateStep(BaseStep):
    """Provision ``daily/<date>/<slug>.md`` (idempotent); refresh day index."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    async def execute(self):
        assert self.context is not None
        slug: str = self.context.get("slug", "") or ""
        day: str = (self.context.get("date") or "").strip() or _date.today().isoformat()

        err = validate_slug(slug)
        if err:
            self._fail(err)
            return None

        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        path_rel = f"{daily_dir}/{day}/{slug}.md"
        path_abs = (self.vault_path / path_rel).resolve()
        existed = path_abs.is_file()

        if not existed:
            post = frontmatter.Post("", name=slug)
            text = frontmatter.dumps(post)
            if not text.endswith("\n"):
                text += "\n"
            try:
                await write_file_safe(path_abs, text, encoding="utf-8")
            except Exception as e:  # pylint: disable=broad-except
                self._fail(f"create failed: {e}", date=day, slug=slug, path=path_rel)
                return None

        index = await refresh_day_index(self.file_store, day, daily_dir)
        payload = {
            "date": day,
            "slug": slug,
            "path": path_rel,
            "created": not existed,
            "index": index,
        }
        self.context.response.success = True
        verb = "Created" if not existed else "Reused existing"
        self.context.response.answer = f"{verb} daily note {path_rel}"
        self.context.response.metadata.update(payload)
        self.logger.info(
            f"[{self.name}] {'created' if not existed else 'reused'} path={path_rel}",
        )
        return self.context.response
