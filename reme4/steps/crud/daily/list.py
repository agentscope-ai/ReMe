"""``daily_list`` — list the notes under a single day (pure read, no side effects).

Returns one row per ``daily/<date>/<slug>.md`` note file with its
vault-relative ``path`` plus ``slug`` / ``name`` / ``description``
from frontmatter. Sorted by slug for stable output.

**Does NOT refresh** ``daily/<date>.md`` — call ``daily_reindex``
explicitly when the index page needs to be rebuilt. Decoupling
read from write keeps each step's effect predictable.

Input is a single optional ``date`` (ISO ``YYYY-MM-DD``); falls back
to today.
"""

from datetime import date as _date
from pathlib import Path

from ._daily_io import scan_notes
from ...base_step import BaseStep

from ....components import R


@R.register("daily_list_step")
class DailyListStep(BaseStep):
    """List the notes under a single day. Pure read — no index refresh."""

    async def execute(self):
        assert self.context is not None
        day: str = (self.context.get("date") or "").strip() or _date.today().isoformat()
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        vault_dir = Path(self.file_store.vault_path or ".").resolve()

        scanned = scan_notes(vault_dir, day, daily_dir)
        notes = [
            {
                "path": n["path"],
                "slug": n["slug"],
                "name": n["name"],
                "description": n["description"],
            }
            for n in scanned
        ]

        self.context.response.success = True
        self.context.response.answer = f"Listed {len(notes)} note(s) for {day}"
        self.context.response.metadata.update({"date": day, "notes": notes})
