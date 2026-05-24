"""``daily_list`` — list the workspaces under a single day.

Always rebuilds the day index ``daily/<date>.md`` as a side effect (the
freshly-rendered workspace inventory is exactly what the caller is asking
to see), then returns one row per ``daily/<date>/<slug>/<slug>.md``
summary with its vault-relative ``path`` plus ``name`` / ``description``
from frontmatter.

Distinct from :mod:`daily_reindex` even though both call
``refresh_day_index``: this one is the read view (consumers want the
workspace inventory), so the index-page bookkeeping fields (``path`` of
``daily/<date>.md``, ``created``) are stripped from the response;
``daily_reindex`` is the write view (consumers want to know what was
rebuilt) and returns those fields without the per-workspace list.

Input is a single optional ``date`` (ISO ``YYYY-MM-DD``); falls back to
today.
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


@R.register("daily_list_step")
class DailyListStep(BaseStep):
    """List the workspaces under a single day; also refreshes ``daily/<date>.md``."""

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
                "workspaces": refreshed["workspaces"],
            },
        )
