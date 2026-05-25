"""``daily_create`` — create a daily note file + refresh the day index.

A daily note is the single file ``daily/<YYYY-MM-DD>/<slug>.md``.
The day-level index ``daily/<YYYY-MM-DD>.md`` is also refreshed so
all index views (navigation, search, distill input) reflect the new
note.

This step bakes the conventions an agent shouldn't have to memorize
on every call:

- path template (``date + slug → daily/<date>/<slug>.md``)
- reserved-field default: ``name`` falls back to ``slug``
- day-index refresh so ``daily/<date>.md`` stays consistent

Frontmatter is intentionally minimal: only the reserved ``name``
field is written by default. Anything else (status, lifecycle, scope,
role, created, ...) is user-defined — supply it via the generic
``property:update`` step after creation.

It does NOT do content R-M-W. Once the note exists, the agent
uses ``file_write`` for body edits and ``property:update`` for
frontmatter tweaks.

Idempotent: when the note file already exists, returns
``{created: False, ...}`` without touching it. The day index is still
refreshed because sibling notes may have changed since the last
call — keeping the index in sync is cheap and self-healing. Pass
``refresh_index=False`` to skip the refresh (rare; mostly for tests /
batch-create flows where the caller will refresh once at the end).
"""

from datetime import date as _date
from pathlib import Path

import frontmatter

from ._day_index import refresh_day_index
from ..base_step import BaseStep

from ...components import R


@R.register("daily_create_step")
class DailyCreateStep(BaseStep):
    """Create the note file ``daily/<date>/<slug>.md`` (idempotent); refresh day index."""

    async def execute(self):
        assert self.context is not None
        slug: str = self.context.get("slug", "") or ""
        body: str = self.context.get("body", "") or ""
        day: str = self.context.get("date") or _date.today().isoformat()
        name: str = self.context.get("name", "") or ""
        refresh_index: bool = bool(self.context.get("refresh_index", True))
        assert slug, "slug is required"

        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        path_rel = f"{daily_dir}/{day}/{slug}.md"
        vault_dir = Path(self.file_store.vault_path or ".")
        path_abs = (vault_dir / path_rel).resolve()

        # Idempotent: existing note returns "already exists" — caller
        # decides whether to edit (file_read + file_write) or skip.
        already_existed = path_abs.is_file()
        if not already_existed:
            path_abs.parent.mkdir(parents=True, exist_ok=True)
            post = frontmatter.Post(body, name=name or slug)
            path_abs.write_text(frontmatter.dumps(post), encoding="utf-8")

        payload: dict = {
            "date": day,
            "slug": slug,
            "path": path_rel,
            "created": not already_existed,
        }

        # Refresh even on the idempotent path: sibling notes may have
        # changed since the last call and the index should track.
        if refresh_index:
            payload["index"] = await refresh_day_index(self.file_store, day, daily_dir)

        self.context.response.success = True
        verb = "Created" if not already_existed else "Reused existing"
        self.context.response.answer = f"{verb} daily note {path_rel}"
        self.context.response.metadata.update(payload)
