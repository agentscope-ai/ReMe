"""``daily_reindex_step`` — rebuild ``daily/<date>.md`` from its workspaces.

The day index ``daily/<date>.md`` is a derived artifact whose job is to
list and describe every workspace under ``daily/<date>/``. It is **not**
auto-refreshed — ``daily_resolve`` (folder genesis), ``file_write`` and
``frontmatter_update`` all leave it stale. This step is the standalone
writer that rebuilds it for batch flows (historical backfill, drift
recovery, end-of-batch consolidation).

The same rebuild also runs as a side effect of :mod:`daily_list`. The
two steps differ in their response: this one is the **write view** —
it reports the index-page path and a ``created`` flag (true when the
file was just emitted for the first time), which is what a caller
running a rebuild wants to confirm. ``daily_list`` is the **read view**
and returns the per-workspace inventory instead.

Input is a single optional ``date`` (ISO ``YYYY-MM-DD``); falls back to
today.

Always idempotent and safe to re-run.
"""

from __future__ import annotations

from datetime import date as _date

from ._day_index import refresh_day_index
from ..base_step import BaseStep
from ...utils import set_answer

from ...components import R
from ...enumeration import ComponentEnum


def _today_iso() -> str:
    return _date.today().isoformat()


@R.register("daily_reindex_step")
class DailyReindexStep(BaseStep):
    """Rebuild ``daily/<date>.md`` from the current state of its workspaces."""

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        day: str = (self.context.get("date") or _today_iso()).strip() or _today_iso()
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        refreshed = await refresh_day_index(self.file_store, day, daily_dir)
        if "error" in refreshed:
            self.context.response.success = False
            set_answer(self.context, refreshed)
            return
        self.context.response.success = True
        set_answer(
            self.context,
            {
                "date": refreshed["date"],
                "path": refreshed["path"],
                "created": refreshed["created"],
                "workspaces_count": len(refreshed["workspaces"]),
            },
        )
