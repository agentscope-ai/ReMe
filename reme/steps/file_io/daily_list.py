"""``daily_list`` — list daily notes with optional date ranges and pagination (pure read).

Returns one row per ``daily/<date>/<name>.md`` note file. The human answer
contains the workspace-relative ``path`` plus every frontmatter key/value.
Response metadata includes ``notes`` as the original list of flat dicts:
``{"path": ..., "name": ..., "description": ..., "session_id": ...,
"source_conversation": ..., ...}``.

**Does NOT refresh** ``daily/<date>.md`` — call ``daily_reindex``
explicitly when the index page needs to be rebuilt. Decoupling
read from write keeps each step's effect predictable.

Without range or pagination parameters, the optional ``date`` defaults to today
and the original response shape is preserved. Ranges are inclusive and ordered
newest day first, then by path. Pagination limits output, not scan I/O.
"""

import asyncio
from pathlib import Path

from ._daily_index import parse_daily_date, scan_notes
from ._path import resolve_path
from ..base_step import BaseStep
from ...components import R
from ...steps.evolve import now


@R.register("daily_list_step")
class DailyListStep(BaseStep):
    """Read daily notes without refreshing indexes or changing workspace files."""

    def _collect_params(self) -> tuple[dict, int | None, int | None]:
        """Keep legacy calls unpaged; range queries opt into bounded pages."""
        assert self.context is not None
        ranged = any(key in self.context for key in ("start_date", "end_date"))
        if ranged:
            if self.context.get("date"):
                raise ValueError("date cannot be combined with start_date/end_date")
            scope = {key: self.context.get(key) for key in ("start_date", "end_date")}
        else:
            tz = self.app_context.app_config.timezone if self.app_context is not None else None
            scope = {"date": self.context.get("date") or now(tz).strftime("%Y-%m-%d")}
        for key, value in scope.items():
            if not isinstance(value, str) or parse_daily_date(value) != value:
                raise ValueError(f"{key} must be YYYY-MM-DD")
        if ranged and scope["start_date"] > scope["end_date"]:
            raise ValueError("start_date must not be later than end_date")
        if not ranged and not any(key in self.context for key in ("page", "page_size")):
            return scope, None, None
        page, page_size = self.context.get("page", 1), self.context.get("page_size", 100)
        if type(page) is not int or page < 1:
            raise ValueError("page must be an integer greater than or equal to 1")
        if type(page_size) is not int or not 1 <= page_size <= 1000:
            raise ValueError("page_size must be an integer between 1 and 1000")
        return scope, page, page_size

    @classmethod
    def _scan(cls, workspace_dir: Path, daily_dir: str, scope: dict, allowed_paths) -> list[dict]:
        """Inspect existing date directories only; permission checks precede file reads."""
        root, err = resolve_path(workspace_dir, daily_dir)
        if err or root is None:
            raise ValueError(err or "invalid daily directory")
        if "date" in scope:
            days = [scope["date"]]
        elif root.is_dir():
            days = sorted(
                (
                    entry.name
                    for entry in root.iterdir()
                    if parse_daily_date(entry.name) == entry.name
                    and scope["start_date"] <= entry.name <= scope["end_date"]
                ),
                reverse=True,
            )
        else:
            days = []
        return [
            cls._project(note)
            for day in days
            for note in scan_notes(workspace_dir, day, daily_dir, allowed_paths=allowed_paths)
        ]

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
        """List authorized notes, preserving the legacy single-day response by default."""
        assert self.context is not None
        try:
            scope, page, page_size = self._collect_params()
            workspace_dir = Path(self.file_store.workspace_path or ".").resolve()
            notes = await asyncio.to_thread(
                self._scan,
                workspace_dir,
                self.config_value("daily_dir"),
                scope,
                self.context.get("_allowed_paths"),
            )
        except (ValueError, OSError, RuntimeError) as exc:
            self.context.response.success = False
            self.context.response.answer = f"Error: {exc}"
            self.context.response.metadata["error"] = str(exc)
            return
        metadata = dict(scope)
        if page is not None and page_size is not None:
            total = len(notes)
            metadata.update(
                {
                    "page": page,
                    "page_size": page_size,
                    "total_notes": total,
                    "total_pages": (total + page_size - 1) // page_size,
                },
            )
            notes = notes[(page - 1) * page_size : page * page_size]
        metadata.update({"count": len(notes), "notes": notes})
        self.context.response.success = True
        lines = [self._format_note_line(note) for note in notes]
        label = scope.get("date") or f"{scope['start_date']} to {scope['end_date']}"
        empty = f"No notes found for {label}"
        if metadata.get("total_notes") and not notes:
            empty = f"No notes on page {page} (total notes: {metadata['total_notes']})"
        answer = "\n".join(lines) if lines else empty
        if page is not None:
            header = " ".join(f"{key}: {metadata[key]}" for key in ("page", "page_size", "total_notes", "total_pages"))
            answer = f"{header}\n{answer}"
        self.context.response.answer = answer
        self.context.response.metadata.update(metadata)
        self.logger.info(f"[{self.name}] scope={scope} notes={len(notes)}")
