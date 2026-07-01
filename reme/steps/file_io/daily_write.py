"""Write a daily markdown note by dispatching the generic ``write`` step."""

from ._daily_index import parse_daily_date, refresh_day_index, validate_session_id
from ._path import validate_filename_component
from ..base_step import BaseStep
from ...components import R
from ...steps.evolve import now

_RESERVED_METADATA_KEYS = {"name", "description", "session_id", "source_conversation"}


@R.register("daily_write_step")
class DailyWriteStep(BaseStep):
    """Write ``daily/<date>/<name>.md`` with conversation frontmatter."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    def _session_dir(self) -> str:
        return str(self.config_value("session_dir")).strip("/")

    def _session_link(self, session_id: str) -> str:
        return f"[[{self._session_dir()}/dialog/{session_id}.jsonl]]"

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

    async def execute(self):
        assert self.context is not None
        collected = self._collect_required()
        if collected is None:
            return None

        name, description, session_id, content = collected
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        raw_date = self.context.get("date", "")
        day = parse_daily_date(raw_date) if raw_date else now(tz).strftime("%Y-%m-%d")
        if raw_date and day is None:
            self._fail("date must be YYYY-MM-DD", date=raw_date)
            return self.context.response
        daily_dir = self.config_value("daily_dir")
        path = f"{daily_dir}/{day}/{name}.md"
        source_conversation = self._session_link(session_id)

        write_responses = await self.dispatch_steps(
            [{"backend": "write_step"}],
            path=path,
            name=name,
            description=description,
            content=content,
            metadata=self._metadata(session_id),
        )
        write_response = write_responses[0]
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

        self.context.response.success = True
        self.context.response.answer = write_answer
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
