"""``daily_create`` — provision a session note under a daily folder.

When ``session_id`` is provided: ``daily/<date>/<session_id>.md``
When ``session_id`` is empty: ``daily/<date>.md`` (the day-level file)

Validates the session_id (when non-empty), mkdirs the day folder,
writes an empty-body note with frontmatter ``{name, description}``
if (and only if) the file does not already exist, refreshes the day
index (only when session_id is non-empty), and returns the
vault-relative path.

Idempotent: when the note already exists this is a no-op write.
The caller fills the body via ``file_write`` / ``file_edit`` /
``file_append`` (or a native editor); ``daily_create`` deliberately
does not accept a body.

Inputs:
    session_id (optional) — the note's session identifier (also the file stem);
               empty string → day-level file
    date       (optional, ``YYYY-MM-DD``; empty = today)

Outputs:
    answer   = one-line human-readable status
    metadata = {date, session_id, path, created, index?}
"""

from pathlib import Path

import frontmatter

from ._daily_index import refresh_day_index, validate_session_id
from ._file_io import write_file_safe
from ..base_step import BaseStep
from ...components import R
from ...steps.evolve import now


@R.register("daily_create_step")
class DailyCreateStep(BaseStep):
    """Provision a daily note (idempotent); refresh day index when applicable."""

    def _fail(self, message: str, **meta) -> None:
        """Mark response failed; copy ``meta`` into ``response.metadata``."""
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    def _collect_params(self) -> tuple[str, str, str]:
        """Read ``session_id`` + ``date`` from context; default ``date`` today, ``daily_dir`` from app config."""
        assert self.context is not None
        session_id = self.context.get("session_id", "")
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        day = self.context.get("date", "") or now(tz).strftime("%Y-%m-%d")
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        return session_id, day, daily_dir

    @staticmethod
    def _empty_note_text(name: str) -> str:
        """Serialize an empty-body markdown note with frontmatter ``{name, description}``; trailing newline."""
        text = frontmatter.dumps(frontmatter.Post("", name=name, description=""))
        return text if text.endswith("\n") else text + "\n"

    async def _create_if_missing(self, path_abs: Path, name: str) -> bool:
        """Write the empty note only when the file is absent. Returns ``True`` iff a new file was created."""
        if path_abs.is_file():
            return False
        await write_file_safe(path_abs, self._empty_note_text(name), encoding="utf-8")
        return True

    def _set_success(self, payload: dict, created: bool) -> None:
        """Stamp the response with success + human-readable answer + metadata payload."""
        assert self.context is not None
        self.context.response.success = True
        self.context.response.answer = f"{'Created' if created else 'Reused existing'} daily note {payload['path']}"
        self.context.response.metadata.update(payload)

    async def execute(self):
        """Provision the note file, optionally refresh the day index, stamp the response."""
        assert self.context is not None
        session_id, day, daily_dir = self._collect_params()

        if session_id:
            err = validate_session_id(session_id)
            if err:
                self._fail(err)
                return None
            path_rel = f"{daily_dir}/{day}/{session_id}.md"
            name = session_id
        else:
            path_rel = f"{daily_dir}/{day}.md"
            name = day

        path_abs = (self.vault_path / path_rel).resolve()
        try:
            created = await self._create_if_missing(path_abs, name)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"create failed: {e}", date=day, session_id=session_id, path=path_rel)
            return None

        payload: dict = {"date": day, "session_id": session_id, "path": path_rel, "created": created}
        if session_id:
            payload["index"] = await refresh_day_index(self.file_store, day, daily_dir)

        self._set_success(payload, created)
        self.logger.info(f"[{self.name}] {'created' if created else 'reused'} path={path_rel}")
        return self.context.response
