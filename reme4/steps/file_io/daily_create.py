"""``daily_create`` — provision a note slug under a daily folder: ``daily/<date>/<slug>.md``.

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
    slug (required, validated) — the note's name (also the file stem)
    date (optional, ``YYYY-MM-DD``; empty = today)

Outputs:
    answer   = one-line human-readable status
    metadata = {date, slug, path, created, index?}
"""

from datetime import date as _date
from pathlib import Path

import frontmatter

from ._file_io import refresh_day_index, validate_slug, write_file_safe
from ..base_step import BaseStep
from ...components import R


@R.register("daily_create_step")
class DailyCreateStep(BaseStep):
    """Provision ``daily/<date>/<slug>.md`` (idempotent); refresh day index."""

    def _fail(self, message: str, **meta) -> None:
        """Mark response failed; copy ``meta`` into ``response.metadata``."""
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    def _collect_params(self) -> tuple[str, str, str]:
        """Read ``slug`` + ``date`` from context; default ``date`` to today (``YYYY-MM-DD``), ``daily_dir`` from app config."""
        assert self.context is not None
        slug = self.context.get("slug", "")
        day = self.context.get("date", "") or _date.today().strftime("%Y-%m-%d")
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        return slug, day, daily_dir

    @staticmethod
    def _empty_note_text(slug: str) -> str:
        """Serialize an empty-body markdown note with frontmatter ``{name: slug}``; trailing newline guaranteed."""
        text = frontmatter.dumps(frontmatter.Post("", name=slug))
        return text if text.endswith("\n") else text + "\n"

    async def _create_if_missing(self, path_abs: Path, slug: str) -> bool:
        """Write the empty note only when the file is absent. Returns ``True`` iff a new file was created."""
        if path_abs.is_file():
            return False
        await write_file_safe(path_abs, self._empty_note_text(slug), encoding="utf-8")
        return True

    def _set_success(self, payload: dict, created: bool) -> None:
        """Stamp the response with success + human-readable answer + metadata payload."""
        assert self.context is not None
        self.context.response.success = True
        self.context.response.answer = f"{'Created' if created else 'Reused existing'} daily note {payload['path']}"
        self.context.response.metadata.update(payload)

    async def execute(self):
        """Validate the slug, provision the note file, refresh the day index, stamp the response."""
        assert self.context is not None
        slug, day, daily_dir = self._collect_params()

        err = validate_slug(slug)
        if err:
            self._fail(err)
            return None

        path_rel = f"{daily_dir}/{day}/{slug}.md"
        path_abs = (self.vault_path / path_rel).resolve()
        try:
            created = await self._create_if_missing(path_abs, slug)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"create failed: {e}", date=day, slug=slug, path=path_rel)
            return None

        index = await refresh_day_index(self.file_store, day, daily_dir)
        self._set_success({"date": day, "slug": slug, "path": path_rel, "created": created, "index": index}, created)
        self.logger.info(f"[{self.name}] {'created' if created else 'reused'} path={path_rel}")
        return self.context.response
