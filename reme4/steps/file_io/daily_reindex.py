"""``daily_reindex_step`` — rebuild ``daily/<date>.md`` from its session notes.

The day index ``daily/<date>.md`` is a derived artifact whose job is to
list and describe every session note file under ``daily/<date>/``. It is
auto-refreshed by ``daily_create``. Generic ops like ``file_write`` /
``file_append`` / ``frontmatter_update`` leave it stale — this step
is the standalone writer to call after batch flows (historical
backfill, drift recovery, end-of-batch consolidation, or a
``frontmatter_update`` that touched ``name`` / ``description``).

This is the **write view**: it reports the index-page path and a
``created`` flag (true when the file was just emitted for the first
time), which is what a caller running a rebuild wants to confirm. For
the per-note inventory use ``daily_list``.

Input is a single optional ``date`` (``YYYY-MM-DD``); falls back to
today.

Always idempotent and safe to re-run.
"""

from ._daily_index import refresh_day_index
from ..base_step import BaseStep
from ...components import R
from ...steps.evolve import now


@R.register("daily_reindex_step")
class DailyReindexStep(BaseStep):
    """Rebuild ``daily/<date>.md`` from the current state of its notes."""

    def _collect_params(self) -> tuple[str, str]:
        """Read ``date`` (default today) and ``daily_dir`` (default ``daily``) from context/app config."""
        assert self.context is not None
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        day = self.context.get("date", "") or now(tz).strftime("%Y-%m-%d")
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        return day, daily_dir

    def _apply_result(self, refreshed: dict) -> None:
        """Mirror the rebuild outcome (surfaced error or success payload) onto the response object."""
        assert self.context is not None
        if "error" in refreshed:
            self.context.response.success = False
            self.context.response.answer = f"Error: {refreshed['error']}"
            self.context.response.metadata.update(refreshed)
            self.logger.info(f"[{self.name}] reindex failed error={refreshed['error']!r}")
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
        self.logger.info(
            f"[{self.name}] date={refreshed['date']} path={refreshed['path']} "
            f"created={refreshed['created']} notes={notes_count}",
        )

    async def execute(self):
        """Trigger the index rebuild and stamp the response."""
        day, daily_dir = self._collect_params()
        self._apply_result(await refresh_day_index(self.file_store, day, daily_dir))
