"""``daily_reindex_step`` — rebuild ``daily/<date>.md`` from its notes.

The day index ``daily/<date>.md`` is a derived artifact whose job is to
list and describe every note file under ``daily/<date>/``. It is auto-
refreshed by ``daily_write`` after every body write. Generic ops like
``file_write`` / ``file_append`` / ``frontmatter_update`` leave it
stale — this step is the standalone writer to call after batch flows
(historical backfill, drift recovery, end-of-batch consolidation, or
a ``frontmatter_update`` that touched ``name`` / ``description``).

This is the **write view**: it reports the index-page path and a
``created`` flag (true when the file was just emitted for the first
time), which is what a caller running a rebuild wants to confirm. For
the per-note inventory use ``daily_list``.

Input is a single optional ``date`` (ISO ``YYYY-MM-DD``); falls back to
today.

Always idempotent and safe to re-run.
"""

from datetime import date as _date

from ._daily_io import refresh_day_index
from ...base_step import BaseStep

from ....components import R


@R.register("daily_reindex_step")
class DailyReindexStep(BaseStep):
    """Rebuild ``daily/<date>.md`` from the current state of its notes."""

    async def execute(self):
        assert self.context is not None
        day: str = (self.context.get("date") or "").strip() or _date.today().isoformat()
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        refreshed = await refresh_day_index(self.file_store, day, daily_dir)
        if "error" in refreshed:
            self.context.response.success = False
            self.context.response.answer = f"Error: {refreshed['error']}"
            self.context.response.metadata.update(refreshed)
            return
        notes_count = len(refreshed["notes"])
        self.context.response.success = True
        self.context.response.answer = f"Reindexed {refreshed['path']} ({notes_count} note(s))"
        self.context.response.metadata.update(
            {
                "date": refreshed["date"],
                "path": refreshed["path"],
                "created": refreshed["created"],
                "notes_count": notes_count,
            },
        )
