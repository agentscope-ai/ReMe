"""Write a daily markdown note by dispatching the generic ``write`` step."""

from pathlib import Path

from ...constants import SESSION_DIALOG_SUBDIR, TRUSTED_SESSION_ID_KWARG

from ._base import _FileIOStep
from ._daily_index import parse_daily_date, refresh_day_index, scan_notes, validate_session_id
from ._path import validate_filename_component
from ...utils.files.paths import normalize_posix_path
from ...core.registry import R
from ...steps.evolve import now

_RESERVED_METADATA_KEYS = {"name", "description", "session_id", "source_conversation"}


@R.register("daily_write_step")
class DailyWriteStep(_FileIOStep):
    """Write a session-linked note, suffixing colliding names without overwriting another session.

    The server injects a trusted session ID for ``auto_memory`` calls. Direct
    tool or MCP callers without that server-owned value are responsible for
    supplying a session ID they are authorized to update.
    """

    def _session_dir(self) -> str:
        return normalize_posix_path(str(self.config_value("session_dir")))

    def _session_link(self, session_id: str) -> str:
        path = normalize_posix_path(f"{self._session_dir()}/{SESSION_DIALOG_SUBDIR}/{session_id}.jsonl")
        return f"[[{path}]]"

    def _collect_required(self) -> tuple[str, str, str, str] | None:
        assert self.context is not None
        missing = [key for key in ("name", "description", "session_id", "content") if self.context.get(key) is None]
        if missing:
            self._fail(f"missing required parameter(s): {', '.join(missing)}")
            return None

        name = str(self.context.get("name")).strip()
        description = str(self.context.get("description")).strip()
        session_id = str(self.context.get("session_id")).strip()
        content = str(self.context.get("content"))

        if err := validate_filename_component(name, kind="name"):
            self._fail(err)
            return None
        if err := validate_session_id(session_id):
            self._fail(err)
            return None
        return name, description, session_id, content

    def _metadata(self, session_id: str) -> dict:
        assert self.context is not None
        metadata_raw = self.context.get("metadata")
        metadata = {}
        if isinstance(metadata_raw, dict):
            metadata = {
                str(k): v for k, v in metadata_raw.items() if k not in _RESERVED_METADATA_KEYS and v is not None
            }
        metadata.update(
            {
                "session_id": session_id,
                "source_conversation": self._session_link(session_id),
            },
        )
        # ``write`` takes name/description as explicit parameters, not metadata.
        return metadata

    def _notes_by_path(self, day: str, daily_dir: str) -> dict[str, dict]:
        """Return each existing daily-note path and its frontmatter metadata."""
        assert self.file_store is not None
        workspace = Path(self.file_store.workspace_path or ".").resolve()
        return {note["path"]: note["metadata"] for note in scan_notes(workspace, day, daily_dir)}

    async def execute(self):  # pylint: disable=too-many-return-statements
        assert self.context is not None
        collected = self._collect_required()
        if collected is None:
            return None

        name, description, session_id, content = collected
        trusted_session_id = self.context.get(TRUSTED_SESSION_ID_KWARG)
        if trusted_session_id is not None:
            trusted_session_id = str(trusted_session_id).strip()
            if err := validate_session_id(trusted_session_id):
                self._fail(f"invalid trusted session_id: {err}")
                return self.context.response
            if session_id != trusted_session_id:
                self._fail("session_id does not match the trusted session", session_id=session_id)
                return self.context.response
            session_id = trusted_session_id

        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        raw_date = self.context.get("date", "")
        day = parse_daily_date(raw_date) if raw_date else now(tz).strftime("%Y-%m-%d")
        if raw_date and day is None:
            self._fail("date must be YYYY-MM-DD", date=raw_date)
            return self.context.response
        daily_dir = self.config_value("daily_dir")
        # A concurrent writer may claim a candidate between scan and create.
        # Exclusive creation below makes that race safe; on conflict, try the
        # next suffix.
        path = f"{daily_dir}/{day}/{name}.md"
        source_conversation = self._session_link(session_id)

        write_response = None
        existing_notes = self._notes_by_path(day, daily_dir)
        for suffix in range(1, 10000):
            path = f"{daily_dir}/{day}/{name if suffix == 1 else f'{name}-{suffix}'}.md"
            existing = existing_notes.get(path)
            if existing is not None:
                if existing.get("session_id") != session_id:
                    continue
                # Rewrites by the owning session keep their existing path.
                exclusive = False
            else:
                exclusive = True

            write_responses = await self.dispatch_steps(
                [{"backend": "write_step"}],
                path=path,
                name=name,
                description=description,
                content=content,
                metadata=self._metadata(session_id),
                _exclusive=exclusive,
            )
            write_response = write_responses[0]
            if write_response is None:
                # Failed Steps mutate the shared RuntimeContext response and
                # return None. An exclusive-create conflict is the one failure
                # that should retry with another name.
                conflict = self.context.response.metadata.pop("conflict", False)
                if conflict:
                    # Another writer claimed this candidate. Refresh ownership
                    # before deciding whether to retry or append a suffix.
                    existing_notes = self._notes_by_path(day, daily_dir)
                    continue
                write_response = self.context.response
            break
        else:
            self._fail("could not allocate a unique daily note path", date=day, name=name)
            return self.context.response

        write_success = write_response.success
        write_answer = write_response.answer
        if not write_success:
            self.context.response.success = False
            self.context.response.answer = f"write failed: {write_answer}"
            self.context.response.metadata.update(
                {"date": day, "path": path, "session_id": session_id, "source_conversation": source_conversation},
            )
            return self.context.response

        index_payload = await refresh_day_index(self.file_store, day, daily_dir)

        self.context.response.success = "error" not in index_payload
        self.context.response.answer = (
            write_answer if self.context.response.success else f"Error: {index_payload['error']}"
        )
        self.context.response.metadata.update(
            {
                "date": day,
                "path": path,
                "name": name,
                "description": description,
                "session_id": session_id,
                "source_conversation": source_conversation,
                "index": index_payload,
            },
        )
        self.logger.info(f"[{self.name}] wrote daily note path={path} session_id={session_id!r}")
        return self.context.response
