"""``daily_list`` — list the notes under a single day (pure read, no side effects).

Returns one row per ``daily/<date>/<session_id>.md`` note file with its
vault-relative ``path``, ``session_id``, and the raw ``metadata`` dict
(full frontmatter). Sorted by session_id for stable output.

**Does NOT refresh** ``daily/<date>.md`` — call ``daily_reindex``
explicitly when the index page needs to be rebuilt. Decoupling
read from write keeps each step's effect predictable.

Input is a single optional ``date`` (``YYYY-MM-DD``); falls back
to today.
"""

from pathlib import Path

from ._daily_index import scan_notes
from ..base_step import BaseStep
from ...components import R
from ...steps.evolve import now


@R.register("daily_list_step")
class DailyListStep(BaseStep):
    """List the notes under a single day. Pure read — no index refresh."""

    def _collect_params(self) -> tuple[str, str, Path]:
        """Read ``date`` (default today, ``YYYY-MM-DD``), resolve ``daily_dir``, locate the vault root on disk."""
        assert self.context is not None
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        day = self.context.get("date", "") or now(tz).strftime("%Y-%m-%d")
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        vault_dir = Path(self.file_store.vault_path or ".").resolve()
        return day, daily_dir, vault_dir

    @staticmethod
    def _project(note: dict) -> dict:
        """Keep only the user-facing keys (drop internal scan_notes fields, if any)."""
        return {"path": note["path"], "session_id": note["session_id"], "metadata": note["metadata"]}

    @staticmethod
    def _format_note_line(note: dict) -> str:
        """Format a single note as ``- path: ... name: ... description: ... <other keys>``."""
        meta: dict = note.get("metadata", {})
        ordered_keys = [k for k in ("name", "description") if k in meta]
        ordered_keys += [k for k in meta if k not in ("name", "description")]
        parts = [f"- path: {note['path']}"]
        for key in ordered_keys:
            value = meta[key]
            if value is None or value == "":
                continue
            value_str = str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            parts.append(f"{key}: {value_str}")
        return " ".join(parts)

    async def execute(self):
        """Scan ``<daily_dir>/<date>/`` and emit one projected record per note."""
        assert self.context is not None
        day, daily_dir, vault_dir = self._collect_params()
        notes = [self._project(n) for n in scan_notes(vault_dir, day, daily_dir)]
        self.context.response.success = True
        lines = [self._format_note_line(n) for n in notes]
        self.context.response.answer = "\n".join(lines) if lines else f"No notes found for {day}"
        self.context.response.metadata.update({"date": day, "count": len(notes)})
        self.logger.info(f"[{self.name}] date={day} notes={len(notes)}")
