"""``daily_list`` — list the notes under a single day (pure read, no side effects).

Returns one row per ``daily/<date>/<name>.md`` note file. The human answer
contains the workspace-relative ``path`` plus every frontmatter key/value.
Response metadata includes ``notes`` as the original list of flat dicts:
``{"path": ..., "name": ..., "description": ..., "session_id": ...,
"source_conversation": ..., ...}``.

**Does NOT refresh** ``daily/<date>.md`` — call ``daily_reindex``
explicitly when the index page needs to be rebuilt. Decoupling
read from write keeps each step's effect predictable.

Input is a single optional ``date`` (``YYYY-MM-DD``); falls back
to today.
"""

from pathlib import Path

from ._daily_index import scan_notes
from ._path import resolve_path
from ..base_step import BaseStep
from ...components import R
from ...steps.evolve import now


@R.register("daily_list_step")
class DailyListStep(BaseStep):
    """List the notes under a single day. Pure read — no index refresh."""

    def _collect_params(self) -> tuple[str, str, Path]:
        """Read ``date`` (default today, ``YYYY-MM-DD``), ``daily_dir``, and the workspace root."""
        assert self.context is not None
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        day = self.context.get("date", "") or now(tz).strftime("%Y-%m-%d")
        daily_dir = self.config_value("daily_dir")
        workspace_dir = Path(self.file_store.workspace_path or ".").resolve()
        return day, daily_dir, workspace_dir

    @staticmethod
    def _project(note: dict) -> dict:
        """Flatten the note path and full frontmatter into one user-facing dict."""
        return {**note["metadata"], "path": note["path"]}

    @staticmethod
    def _format_note_line(note: dict) -> str:
        """Format a note as ``- path: ... name: ... description: ... <other frontmatter>``."""
        ordered_keys = [k for k in ("name", "description", "session_id", "source_conversation") if k in note]
        ordered_keys += [k for k in note if k not in {"path", *ordered_keys}]
        parts = [f"- path: {note['path']}"]
        for key in ordered_keys:
            value = note[key]
            if value is None or value == "":
                continue
            value_str = str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            parts.append(f"{key}: {value_str}")
        return " ".join(parts)

    async def execute(self):
        """Scan ``<daily_dir>/<date>/`` and emit one projected record per note."""
        assert self.context is not None
        day, daily_dir, workspace_dir = self._collect_params()
        _target_dir, err = resolve_path(workspace_dir, f"{daily_dir}/{day}")
        if err:
            self.context.response.success = False
            self.context.response.answer = f"Error: {err}"
            self.context.response.metadata.update({"date": day, "error": err})
            self.logger.info(f"[{self.name}] date={day} error={err!r}")
            return
        notes = [self._project(n) for n in scan_notes(workspace_dir, day, daily_dir)]
        self.context.response.success = True
        lines = [self._format_note_line(n) for n in notes]
        self.context.response.answer = "\n".join(lines) if lines else f"No notes found for {day}"
        self.context.response.metadata.update({"date": day, "count": len(notes), "notes": notes})
        self.logger.info(f"[{self.name}] date={day} notes={len(notes)}")
