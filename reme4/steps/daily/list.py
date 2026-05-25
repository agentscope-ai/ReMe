"""``daily_list`` — list the notes under a single day.

Always rebuilds the day index ``daily/<date>.md`` as a side effect (the
freshly-rendered note inventory is exactly what the caller is asking
to see), then returns one row per ``daily/<date>/<slug>.md`` note
file with its vault-relative ``path`` plus ``name`` / ``description``
from frontmatter.

Distinct from :mod:`daily_reindex` even though both call
``refresh_day_index``: this one is the read view (consumers want the
note inventory), so the index-page bookkeeping fields (``path`` of
``daily/<date>.md``, ``created``) are stripped from the response;
``daily_reindex`` is the write view (consumers want to know what was
rebuilt) and returns those fields without the per-note list.

Input is a single optional ``date`` (ISO ``YYYY-MM-DD``); falls back to
today.
"""

from datetime import date as _date

from ._day_index import refresh_day_index
from ..base_step import BaseStep

from ...components import R


@R.register("daily_list_step")
class DailyListStep(BaseStep):
    """List the notes under a single day; also refreshes ``daily/<date>.md``."""

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
        notes = refreshed["notes"]
        self.context.response.success = True
        self.context.response.answer = f"Listed {len(notes)} note(s) for {refreshed['date']}"
        self.context.response.metadata.update({"date": refreshed["date"], "notes": notes})
