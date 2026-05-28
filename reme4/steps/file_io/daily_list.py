"""``daily_list`` — list the notes under a single day (pure read, no side effects).

Returns one row per ``daily/<date>/<slug>.md`` note file with its
vault-relative ``path``, ``slug``, and the raw ``metadata`` dict
(full frontmatter). Sorted by slug for stable output.

**Does NOT refresh** ``daily/<date>.md`` — call ``daily_reindex``
explicitly when the index page needs to be rebuilt. Decoupling
read from write keeps each step's effect predictable.

Input is a single optional ``date`` (``YYYY-MM-DD``); falls back
to today.
"""

from datetime import date as _date
from pathlib import Path

from ._file_io import scan_notes
from ..base_step import BaseStep
from ...components import R


@R.register("daily_list_step")
class DailyListStep(BaseStep):
    """List the notes under a single day. Pure read — no index refresh."""

    def _collect_params(self) -> tuple[str, str, Path]:
        """Read ``date`` (default today, ``YYYY-MM-DD``), resolve ``daily_dir``, locate the vault root on disk."""
        assert self.context is not None
        day = self.context.get("date", "") or _date.today().strftime("%Y-%m-%d")
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        vault_dir = Path(self.file_store.vault_path or ".").resolve()
        return day, daily_dir, vault_dir

    @staticmethod
    def _project(note: dict) -> dict:
        """Keep only the user-facing keys (drop internal scan_notes fields, if any)."""
        return {"path": note["path"], "slug": note["slug"], "metadata": note["metadata"]}

    async def execute(self):
        """Scan ``<daily_dir>/<date>/`` and emit one projected record per note."""
        assert self.context is not None
        day, daily_dir, vault_dir = self._collect_params()
        notes = [self._project(n) for n in scan_notes(vault_dir, day, daily_dir)]
        self.context.response.success = True
        self.context.response.answer = f"Listed {len(notes)} note(s) for {day}"
        self.context.response.metadata.update({"date": day, "notes": notes})
