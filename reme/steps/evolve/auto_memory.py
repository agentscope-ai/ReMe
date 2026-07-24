"""auto_memory — record conversation facts into a daily note via an agent."""

from pathlib import Path

import aiofiles
import frontmatter
from agentscope.message import Msg

from ._evolve import agent_reply_result_text, format_history, now
from ..base_step import BaseStep
from ..file_io import extract_daily_date, parse_daily_date, refresh_day_index
from ..file_io import validate_filename_component, validate_session_id
from ...components import R

_SESSION_ID_KEY = "session_id"
_SOURCE_CONVERSATION_KEY = "source_conversation"
_MESSAGE_TIME_ALIASES = ("time_created", "timestamp", "createdAt", "timeCreated", "created_time")


def _sanitize_msg_for_save(msg: Msg) -> Msg:
    new_content = []
    changed = False
    for block in msg.content:
        # Tool results often contain recalled memory/search/read output. Keeping
        # them in saved conversation history lets retrieved facts masquerade as
        # user-provided context in future auto-memory runs.
        if block.type == "tool_result":
            changed = True
            continue
        if block.type == "data" and hasattr(block, "source") and getattr(block.source, "type", None) == "base64":
            changed = True
            continue
        new_content.append(block)
    if not changed:
        return msg
    return msg.model_copy(update={"content": new_content})


def _normalize_msg_timestamp(item: dict) -> dict:
    """Map common message timestamp aliases to AgentScope's ``created_at`` field."""
    if item.get("created_at"):
        return item

    for key in _MESSAGE_TIME_ALIASES:
        value = item.get(key)
        if value:
            return {**item, "created_at": value}

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in _MESSAGE_TIME_ALIASES:
            value = metadata.get(key)
            if value:
                return {**item, "created_at": value}

    return item


@R.register("auto_memory_step")
class AutoMemoryStep(BaseStep):
    """Record conversation facts into a daily note via an Agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_tools: list[str] = ["daily_write"]
        self.update_tools: list[str] = ["read_daily", "edit_daily", "frontmatter_update", "write_daily"]

    def _session_dir(self) -> str:
        return str(self.config_value("session_dir")).strip("/")

    def _session_path(self, session_id: str) -> Path:
        return self.file_store.workspace_path / self._session_dir() / "dialog" / f"{session_id}.jsonl"

    def _session_link(self, session_id: str) -> str:
        return f"[[{self._session_dir()}/dialog/{session_id}.jsonl]]"

    def _daily_note_path(self, day: str, name: str) -> str:
        return f"{self.config_value('daily_dir')}/{day}/{name}.md"

    def _frontmatter(self, path: str) -> dict:
        post = frontmatter.loads((self.file_store.workspace_path / path).read_text(encoding="utf-8"))
        return dict(post.metadata or {})

    def _note_bytes(self, path: str) -> bytes | None:
        note_path = self.file_store.workspace_path / path
        if not note_path.is_file():
            return None
        return note_path.read_bytes()

    def _note_modified(self, before_path: str, before_bytes: bytes | None, after_path: str) -> bool:
        if not after_path:
            return False
        after_bytes = self._note_bytes(after_path)
        if after_bytes is None:
            return before_bytes is not None
        return after_path != before_path or before_bytes != after_bytes

    def _find_session_note(self, notes: list[dict], session_id: str) -> dict | None:
        source = self._session_link(session_id)
        for note in notes:
            if str(note.get(_SESSION_ID_KEY, "")).strip() == session_id:
                return note
        for note in notes:
            if str(note.get(_SOURCE_CONVERSATION_KEY, "")).strip() == source:
                return note
        return None

    async def _list_session_note(self, day: str, session_id: str) -> dict | None:
        list_response = await self.run_job("daily_list", date=day)
        if not list_response.success:
            raise RuntimeError(f"daily_list failed: {list_response.answer}")
        notes = list_response.metadata.get("notes") or []
        return self._find_session_note(notes, session_id)

    async def _ensure_session_frontmatter(self, path: str, session_id: str) -> None:
        metadata = {
            _SESSION_ID_KEY: session_id,
            _SOURCE_CONVERSATION_KEY: self._session_link(session_id),
        }
        current = self._frontmatter(path)
        if all(current.get(key) == value for key, value in metadata.items()):
            return
        response = await self.run_job(
            "frontmatter_update",
            path=path,
            metadata=metadata,
        )
        if not response.success:
            raise RuntimeError(f"frontmatter_update failed: {response.answer}")

    async def _rename_from_frontmatter_name(self, path: str, day: str) -> str:
        meta = self._frontmatter(path)
        name = str(meta.get("name", "")).strip()
        if not name:
            return path
        if err := validate_filename_component(name, kind="name"):
            raise RuntimeError(err)

        target_path = self._daily_note_path(day, name)
        if target_path == path:
            return path

        move_response = await self.run_job(
            "move",
            src_path=path,
            dst_path=target_path,
            overwrite=False,
            retarget=True,
        )
        if not move_response.success:
            raise RuntimeError(f"move failed: {move_response.answer}")
        return target_path

    async def _save_session_messages(self, session_id: str, messages: list[Msg]) -> None:
        if not session_id or not messages:
            return

        path = self._session_path(session_id)
        self.logger.info(
            f"[{self.name}] save session start session_id={session_id!r} messages={len(messages)} path={path}",
        )

        existing: list[Msg] = []
        if path.exists():
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        existing.append(Msg.model_validate_json(line))
                    except Exception:
                        pass

        by_id: dict[str, Msg] = {}
        for msg in existing:
            by_id[msg.id] = msg
        for msg in messages:
            by_id[msg.id] = msg
        merged = sorted(by_id.values(), key=lambda m: m.created_at)

        can_append = 0 < len(existing) <= len(merged) and all(
            merged[i].id == existing[i].id for i in range(len(existing))
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        if can_append:
            new_msgs = merged[len(existing) :]
            if new_msgs:
                async with aiofiles.open(path, "a", encoding="utf-8") as f:
                    for msg in new_msgs:
                        await f.write(_sanitize_msg_for_save(msg).model_dump_json() + "\n")
                self.logger.info(
                    f"[{self.name}] save session appended session_id={session_id!r} "
                    f"existing={len(existing)} appended={len(new_msgs)} total={len(merged)}",
                )
            else:
                self.logger.info(
                    f"[{self.name}] save session unchanged session_id={session_id!r} "
                    f"existing={len(existing)} total={len(merged)}",
                )
        else:
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                for msg in merged:
                    await f.write(_sanitize_msg_for_save(msg).model_dump_json() + "\n")
            self.logger.info(
                f"[{self.name}] save session rewrote session_id={session_id!r} "
                f"existing={len(existing)} total={len(merged)}",
            )

    @staticmethod
    def _to_msg(item) -> Msg:
        if isinstance(item, Msg):
            return item
        if isinstance(item, dict):
            item = _normalize_msg_timestamp(item)
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            item = {**item, "content": [{"type": "text", "text": item["content"]}]}
        return Msg.model_validate(item)

    @staticmethod
    def _messages_day(messages: list[Msg]) -> str | None:
        days = [day for msg in messages if (day := extract_daily_date(msg.created_at))]
        return max(days) if days else None

    def _build_messages(self, raw_messages: list) -> list[Msg]:
        """Convert raw message payloads into ``Msg`` objects.

        Overridable hook: subclasses can preprocess ``raw_messages`` (e.g. fill
        in missing timestamps) before conversion.
        """
        return [self._to_msg(item) for item in raw_messages]

    def _reply_extra_kwargs(self, day: str) -> dict:  # pylint: disable=unused-argument
        """Extra keyword arguments for ``agent_wrapper.reply``.

        Overridable hook: subclasses can inject additional reply options such
        as per-tool defaults keyed on ``day``.
        """
        return {}

    # pylint: disable=too-many-return-statements
    async def execute(self):
        assert self.context is not None
        raw_messages = self.context.get("messages") or []
        session_id: str = self.context.get("session_id", "")
        memory_hint: str = self.context.get("memory_hint", "")
        raw_date = self.context.get("date", "")
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        current = now(tz)

        messages: list[Msg] = self._build_messages(raw_messages)
        self.logger.info(
            f"[{self.name}] start session_id={session_id!r} raw_messages={len(raw_messages)} "
            f"messages={len(messages)} hint={bool(memory_hint)}",
        )

        if session_id and (err := validate_session_id(session_id)):
            self.context.response.success = False
            self.context.response.answer = f"Error: {err}"
            self.logger.warning(f"[{self.name}] invalid session_id={session_id!r} err={err}")
            return
        if not session_id:
            self.context.response.success = False
            self.context.response.answer = "Error: session_id is required"
            self.logger.warning(f"[{self.name}] missing session_id")
            return

        day = parse_daily_date(raw_date) if raw_date else self._messages_day(messages) or current.strftime("%Y-%m-%d")
        if raw_date and day is None:
            self.context.response.success = False
            self.context.response.answer = "Error: date must be YYYY-MM-DD"
            self.context.response.metadata.update({"date": raw_date, "modified": False, "n_messages": len(messages)})
            self.logger.warning(f"[{self.name}] invalid date={raw_date!r}")
            return

        await self._save_session_messages(session_id, messages)

        if not messages:
            self.context.response.success = True
            self.context.response.answer = "Skipped: no messages"
            self.context.response.metadata.update({"date": day, "modified": False, "n_messages": 0})
            self.logger.info(f"[{self.name}] Skipped: no messages session_id={session_id!r} modified=False")
            return

        try:
            note = await self._list_session_note(day, session_id)
        except RuntimeError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            self.context.response.metadata.update({"date": day, "modified": False, "n_messages": len(messages)})
            self.logger.info(f"[{self.name}] list failed session_id={session_id!r} answer={str(exc)!r}")
            return

        note_path = str(note["path"]) if note else ""
        created = note is None
        before_note_path = note_path
        before_note_bytes = self._note_bytes(note_path) if note_path else None
        self.logger.info(
            f"[{self.name}] note lookup session_id={session_id!r} path={note_path!r} "
            f"created={created} msgs={len(messages)} hint={bool(memory_hint)}",
        )
        template_key = "user_message_create" if created else "user_message_update"
        user_message = self.prompt_format(
            template_key,
            today=day,
            note=memory_hint or "(none)",
            note_path=note_path,
            session_id=session_id,
            history=format_history(messages),
        )

        self.logger.info(f"[{self.name}] agent start path={note_path} template={template_key}")
        result = await self.agent_wrapper.reply(
            user_message,
            system_prompt=self.prompt_format("system_prompt"),
            job_tools=self.create_tools if created else self.update_tools,
            **self._reply_extra_kwargs(day),
        )
        self.logger.info(f"[{self.name}] agent done path={note_path} has_result={bool(result.get('result'))}")

        if created:
            try:
                note = await self._list_session_note(day, session_id)
            except RuntimeError as exc:
                self.context.response.success = False
                self.context.response.answer = str(exc)
                self.context.response.metadata.update(
                    {"date": day, "path": None, "created": created, "modified": False, "n_messages": len(messages)},
                )
                self.logger.info(f"[{self.name}] post-create list failed session_id={session_id!r} answer={str(exc)!r}")
                return
            if note is None:
                self.context.response.success = True
                self.context.response.answer = agent_reply_result_text(result)
                self.context.response.metadata.update(
                    {"date": day, "path": None, "created": False, "modified": False, "n_messages": len(messages)},
                )
                self.logger.info(f"[{self.name}] done without note session_id={session_id!r} modified=False")
                return
            note_path = str(note["path"])
        else:
            try:
                await self._ensure_session_frontmatter(note_path, session_id)
                note_path = await self._rename_from_frontmatter_name(note_path, day)
            except RuntimeError as exc:
                self.context.response.success = False
                self.context.response.answer = str(exc)
                self.context.response.metadata.update(
                    {
                        "date": day,
                        "path": note_path,
                        "created": created,
                        "modified": self._note_modified(before_note_path, before_note_bytes, note_path),
                        "n_messages": len(messages),
                    },
                )
                self.logger.info(f"[{self.name}] post-update failed path={note_path} answer={str(exc)!r}")
                return

        modified = self._note_modified(before_note_path, before_note_bytes, note_path)
        daily_dir = self.config_value("daily_dir")
        self.logger.info(f"[{self.name}] refresh index start date={day} daily_dir={daily_dir}")
        index_payload = await refresh_day_index(self.file_store, day, daily_dir)
        self.logger.info(f"[{self.name}] refresh index done path={note_path}")

        source_conversation = self._session_link(session_id)
        self.context.response.success = True
        self.context.response.answer = agent_reply_result_text(result)
        self.context.response.metadata.update(
            {
                "date": day,
                "path": note_path,
                "created": created,
                "modified": modified,
                "n_messages": len(messages),
                "source_conversation": source_conversation,
                "index": index_payload,
            },
        )
        self.logger.info(f"[{self.name}] done {note_path} modified={modified}")
