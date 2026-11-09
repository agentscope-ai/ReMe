"""Application-owned, bounded notifications for the selected PA conversation."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import threading
from uuid import uuid4
from zoneinfo import ZoneInfo

from .agent_wrapper import CodexAgentWrapper
from .base_component import BaseComponent
from .component_registry import R
from ._pa_hints import ProactiveHints
from ._pa_reminders import ReminderDelivery
from ._pa_tasks import TaskFiles
from ._pa_inbox import ProactiveInbox
from ._pa_temp_tasks import TempTasks
from ..schema.agent_session import SessionMessage, SessionSnapshot, SessionStatus
from ..schema.pa_event import PaEvent
from ..schema.agent_activity import AgentActivity
from ..schema.pa_session import PaSessionCursor
from ..schema.pa_proactive import PaNotificationWindow, PaProactiveConfig
from ..schema.pa_hint import HintPeriod
from ..schema.reminder import ReminderKind
from ..schema.pa_task import PaTask, PaTaskRun
from ..utils.async_utils import complete_in_thread, finish_cleanup
from ..constants import DEFAULT_MAX_FILE_BYTES

_CURSOR = re.compile(r"([0-9a-f]{32}):(0|[1-9][0-9]{0,19})\Z")
_SESSION_LIMIT = 100
_SESSION_FILE_BYTES = 128 * 1024


@dataclass(slots=True)
class ProactiveRound:
    """One live agent round; deferred reminders outlive this object."""

    session_id: str | None
    id: str = field(default_factory=lambda: uuid4().hex)
    phase: str | None = None
    hint_period: HintPeriod | None = None
    request: tuple[str, ReminderKind] | None = None
    result: dict = field(default_factory=lambda: {"sent": False, "status": "skipped"})
    feedback_version: str | None = None
    history_version: tuple[int, int, int] | None = None


@R.register("local")
class PaRuntime(BaseComponent):
    """Keep notifications rebuildable from session files, without per-client queues.

    Synchronous publication and cursor checks run on the application's event
    loop. A slow reader falls behind the shared buffer and receives a fresh
    snapshot. Existing PA steps share the same application-lifetime state.
    """

    component_type = "pa"

    def __init__(
        self,
        agent_wrapper: str = "default",
        max_events: int = 512,
        max_buffer_bytes: int = 4 * 1024 * 1024,
        heartbeat_seconds: float = 15,
        default_tasks: list[dict] | None = None,
        default_proactive: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        for name, value in (("max_events", max_events), ("max_buffer_bytes", max_buffer_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(heartbeat_seconds, bool)
            or not isinstance(heartbeat_seconds, (int, float))
            or not math.isfinite(heartbeat_seconds)
            or heartbeat_seconds <= 0
        ):
            raise ValueError("heartbeat_seconds must be positive and finite")
        self.agent_wrapper = self.bind(agent_wrapper, CodexAgentWrapper, optional=False)
        self.max_events = max_events
        self.max_buffer_bytes = max_buffer_bytes
        self.heartbeat_seconds = heartbeat_seconds
        self.default_tasks = [PaTask.model_validate(item) for item in default_tasks or []]
        self.default_proactive = (
            PaProactiveConfig.model_validate(default_proactive) if default_proactive is not None else None
        )
        self.session_id: str | None = None
        self.started_at: datetime | None = None
        self.session_lock = asyncio.Lock()
        self.memory_lock = asyncio.Lock()
        self.current_round: ProactiveRound | None = None
        self.hints = ProactiveHints(self)
        self.tasks = TaskFiles(self)
        self.reminders = ReminderDelivery(self)
        self.inbox = ProactiveInbox(self)
        self.temp_tasks = TempTasks(self)
        self.notifications: PaNotificationWindow | None = None
        self._proactive_lock = asyncio.Lock()
        self._work_context_lock = threading.Lock()
        self._sessions_lock = asyncio.Lock()
        self._conversation_lock = asyncio.Lock()
        self._epoch = uuid4().hex
        self._sequence = 0
        self._revision = 0
        self._buffer: deque[tuple[int, str, int]] = deque()
        self._buffer_bytes = 0
        self._changed = asyncio.Event()
        self._closed = True
        self._unsubscribe: Callable[[], None] | None = None
        self._unsubscribe_activity: Callable[[], None] | None = None
        self._native_status: SessionStatus | None = None
        self._activities: dict[str, AgentActivity] = {}
        self._task_activities: dict[str, AgentActivity] = {}
        self._published_status: tuple | None = None
        self._fingerprints: dict[str, bytes] = {}
        self._needs_refresh = False
        self._reset_reason = "cursor_expired"

    @property
    def proactive_inbox_path(self) -> Path:
        """Keep proactive feedback in the PA component's metadata directory."""
        return self._metadata_file("proactive_inbox.md")

    def _metadata_file(self, name: str) -> Path:
        path = self.component_metadata_path / name
        if not path.resolve().is_relative_to(self.workspace_path.resolve()) or path.resolve() != path.absolute():
            raise ValueError("PA metadata must stay inside the workspace without symlinks")
        return path

    def _now(self) -> datetime:
        timezone = self.app_context.app_config.timezone if self.app_context else None
        return datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()

    @property
    def proactive_config_path(self) -> Path:
        """Return the user-owned wakeup configuration path."""
        return self._metadata_file("proactive.json")

    def _load_proactive_config(self) -> PaProactiveConfig:
        with self.proactive_config_path.open("rb") as handle:
            content = handle.read(4097)
        if len(content) > 4096:
            raise ValueError("proactive.json exceeds 4096 bytes")
        return PaProactiveConfig.model_validate_json(content)

    async def get_proactive_config(self) -> PaProactiveConfig:
        """Reload user edits without caching a second configuration source."""
        try:
            return await asyncio.to_thread(self._load_proactive_config)
        except (OSError, ValueError):
            self.notifications = None
            self.publish_status()
            raise

    async def proactive_skip_reason(self, config: PaProactiveConfig | None = None) -> str | None:
        """Apply local scheduling rules before spending any model tokens."""
        config = config or await self.get_proactive_config()
        window = await self.reminders.window(config)
        if not config.enabled:
            return "disabled"
        if not config.is_active(self._now()):
            return "outside_active_hours"
        return "day_closed" if window.closing_message_id is not None else None

    @property
    def proactive_context_path(self) -> Path:
        """Consumed inputs and partial progress, retained until the round completes."""
        return self._metadata_file("proactive_context.jsonl")

    @property
    def proactive_todo_path(self) -> Path:
        """Keep unfinished proactive work beside the other PA state files."""
        return self._metadata_file("proactive_todo.md")

    async def read_proactive_todo(self) -> str:
        """Load the latest bounded TODO, including after an application restart."""
        try:
            return await asyncio.to_thread(self.tasks.read, self.proactive_todo_path)
        except FileNotFoundError:
            return ""

    async def save_proactive_todo(self, content: str) -> None:
        """Publish the model's TODO list only after its final query completes."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Proactive TODO list must not be empty")
        content = content.strip() + "\n"
        if len(content.encode("utf-8")) > DEFAULT_MAX_FILE_BYTES:
            raise ValueError("Proactive TODO exceeds the file read limit")
        await complete_in_thread(self._write_text, self.proactive_todo_path, content)

    def _checkpoint_context(self, round_id: str | None, kind: str, value) -> None:
        if round_id is not None and value:
            with self._work_context_lock:
                self._append_jsonl(
                    self.proactive_context_path,
                    {
                        "round_id": round_id,
                        "time": self._now().isoformat(),
                        "kind": kind,
                        "value": value,
                    },
                )

    async def checkpoint_proactive(self, round_id: str | None, kind: str, value) -> None:
        """Durably retain consumed inputs or progress for interrupted rounds."""
        await complete_in_thread(self._checkpoint_context, round_id, kind, value)

    async def clear_proactive_context(self) -> None:
        """Clear recovery records after the work queries complete."""

        def clear():
            with self._work_context_lock:
                self._write_text(self.proactive_context_path, "")

        await complete_in_thread(clear)

    async def write_proactive_inbox(self, message: str) -> dict:
        """Queue a message for the proactive agent's next wakeup."""
        return await finish_cleanup(self.inbox.write(message))

    async def read_proactive_inbox(
        self,
        max_bytes: int = DEFAULT_MAX_FILE_BYTES,
        *,
        round_id: str | None = None,
    ) -> dict:
        """Checkpoint newly appended or edited Markdown without deleting the user's inbox."""
        return await self.inbox.read(round_id=round_id, max_bytes=max_bytes)

    @property
    def chat_inbox_path(self) -> Path:
        """Append-only background messages and delivery states addressed to Main."""
        return self._metadata_file("chat_inbox.jsonl")

    def _append_jsonl(self, path: Path, data: dict) -> None:
        self._append_text(path, json.dumps(data, ensure_ascii=False) + "\n")

    def _append_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags, 0o600), "a", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

    @asynccontextmanager
    async def proactive_round(self):
        """Serialize proactive rounds and bind their tool-call identity."""
        async with self._proactive_lock:
            current = ProactiveRound(session_id=self.session_id)
            current.feedback_version = await asyncio.to_thread(self.inbox.version)
            self.current_round = current
            try:
                yield current
            finally:
                self.current_round = None

    async def send_chat_message(
        self,
        content: str,
        kind: ReminderKind,
        *,
        round_id: str,
        session_id: str | None,
    ) -> dict:
        """Own binding, validation, one-decision semantics and cancellation-safe publication."""
        return await finish_cleanup(self._submit_chat_message(content, kind, round_id, session_id))

    async def _submit_chat_message(
        self,
        content: str,
        kind: ReminderKind,
        round_id: str,
        session_id: str | None,
    ) -> dict:
        async with self.reminders.lock:
            current = self.current_round
            if current is None or current.id != round_id or current.session_id != session_id:
                raise ValueError("send_chat_message requires a live, server-bound proactive round")
            if current.phase != "query_message":
                raise ValueError("send_chat_message is only allowed during query_message")
            if not isinstance(content, str) or len(content.strip()) > 4000:
                raise ValueError("content must be a string of at most 4000 characters")
            content = content.strip()
            if kind not in ("normal", "morning", "midday", "evening"):
                raise ValueError("kind must be normal, morning, midday or evening")
            if not content and kind != "normal":
                raise ValueError("Empty decisions must use kind=normal")
            if current.request is not None:
                if current.request != (content, kind):
                    raise ValueError("Only one reminder decision may be submitted per round")
                if not content:
                    return current.result
            if content and current.request is None:
                self.reminders.check_history(current.history_version)
            result = await self.reminders.submit(content, kind, session_id=session_id, message_id=current.id)
            current.request = (content, kind)
            current.result = result
            return result

    async def deliver_pending_reminder(self) -> None:
        """Check deferred delivery without starting a model turn."""
        await self.reminders.deliver_pending()

    async def get_chat_messages(
        self,
        limit: int = 10,
        offset: int = 0,
        *,
        round_id: str | None = None,
        cron_run_id: str | None = None,
    ) -> list[dict]:
        """Read history and bind its version to the caller's current decision."""
        records, version = await self.reminders.history(limit, offset)
        current = None
        if round_id is not None:
            current = self.current_round
            if current is None or current.id != round_id:
                raise ValueError("History requires a live proactive round")
        elif cron_run_id is not None:
            current = self.tasks.message_context(cron_run_id)
        if current is not None:
            if offset and current.history_version != version:
                raise ValueError("Reminder history changed; call get_chat_messages from offset=0 again")
            current.history_version = version
        return records

    @staticmethod
    def _reverse_jsonl_lines(handle):
        """Scan JSONL from EOF without loading its entire history."""
        position = handle.seek(0, os.SEEK_END)
        pending = b""
        while position:
            size = min(position, 8192)
            position -= size
            handle.seek(position)
            lines = (handle.read(size) + pending).split(b"\n")
            pending = lines[0]
            for line in reversed(lines[1:]):
                if line.strip():
                    yield line
        if pending.strip():
            yield pending

    @property
    def cron_path(self) -> Path:
        """User-editable cron definitions."""
        return self._metadata_file("cron.json")

    @property
    def cron_runs_path(self) -> Path:
        """Append-only cron start and finish records."""
        return self._metadata_file("cron_runs.jsonl")

    @property
    def session_ids_path(self) -> Path:
        """Recent chat sessions, oldest first, with persisted read cursors."""
        return self._metadata_file("session_ids.json")

    def _load_session_ids(self) -> list[PaSessionCursor]:
        try:
            with self.session_ids_path.open("rb") as handle:
                content = handle.read(_SESSION_FILE_BYTES + 1)
        except FileNotFoundError:
            return []
        if len(content) > _SESSION_FILE_BYTES:
            raise ValueError("session_ids.json exceeds 128 KiB")
        data = json.loads(content)
        if not isinstance(data, list) or len(data) > _SESSION_LIMIT:
            raise ValueError("session_ids.json must contain at most 100 session records")
        sessions = [PaSessionCursor.model_validate(item) for item in data]
        if len({item.session_id for item in sessions}) != len(sessions):
            raise ValueError("session_ids.json contains duplicate sessions")
        return sessions

    async def remember_session(self, session_id: str) -> None:
        """Move a selected chat to the end without discarding its read cursor."""
        async with self._sessions_lock:
            sessions = await asyncio.to_thread(self._load_session_ids)
            current = next((item for item in sessions if item.session_id == session_id), None)
            active_at = self._now().isoformat()
            if current is None:
                current = PaSessionCursor(session_id=session_id, last_active_at=active_at)
            current.last_active_at = active_at
            sessions = [item for item in sessions if item.session_id != session_id] + [current]
            await complete_in_thread(
                self._write_json,
                self.session_ids_path,
                [item.model_dump() for item in sessions[-_SESSION_LIMIT:]],
            )

    async def read_conversation(
        self,
        max_turns: int = 10,
        max_message_chars: int = 2000,
        max_chars: int = 20000,
        *,
        round_id: str | None = None,
    ) -> str:
        """Consume recent unread chat rounds, keeping each user input with its replies."""
        for name, value, maximum in (
            ("max_turns", max_turns, 100),
            ("max_message_chars", max_message_chars, 32000),
            ("max_chars", max_chars, 32000),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 1 to {maximum}")

        def timestamp(value: str) -> float:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except (ValueError, OverflowError, OSError):
                return 0.0

        def truncate(text: str, size: int) -> str:
            marker = "…[已截断]"
            return text if len(text) <= size else text[: max(0, size - len(marker))] + marker[:size]

        async with self._conversation_lock:
            async with self._sessions_lock:
                sessions = await asyncio.to_thread(self._load_session_ids)
            candidates, cursors = [], {}
            for session_index, entry in enumerate(sessions):
                snapshot = await self.agent_wrapper.read_session(entry.session_id)
                positions = {message.id: index for index, message in enumerate(snapshot.messages)}
                if entry.last_read_message_id is not None and entry.last_read_message_id not in positions:
                    raise ValueError(f"Conversation cursor is no longer present in session {entry.session_id}")
                last_position = positions.get(entry.last_read_message_id, -1)
                turns = []
                for index, message in enumerate(snapshot.messages):
                    if message.source != "chat" or not message.content.strip():
                        continue
                    if message.role == "user" or not turns:
                        turns.append([])
                    turns[-1].append((index, message))
                for turn_index, turn in enumerate(turns):
                    if turn[-1][0] <= last_position:
                        continue
                    if any(not message.completed for _, message in turn):
                        break
                    if snapshot.state.status == "active" and turn_index == len(turns) - 1:
                        break
                    messages = [message for _, message in turn]
                    created_at = next((message.created_at for message in messages if message.created_at), None)
                    candidates.append(
                        (
                            timestamp(created_at or entry.last_active_at),
                            session_index,
                            turn_index,
                            entry.session_id,
                            messages,
                        ),
                    )
                    cursors[entry.session_id] = messages[-1].id
                candidates = sorted(candidates, key=lambda item: item[:3])[-max_turns:]
            chunks, remaining = [], max_chars
            for _time, _session_index, _turn_index, session_id, messages in reversed(candidates):
                headers = [
                    f"[{message.created_at or '时间未知'}][session={session_id}][{message.role}]\n"
                    for message in messages
                ]
                separator_size = 2 * (len(messages) - 1) + (2 if chunks else 0)
                budget = min(max_message_chars, (remaining - sum(map(len, headers)) - separator_size) // len(messages))
                if budget < 1:
                    break
                text = "\n\n".join(
                    header + truncate(message.content.strip(), budget) for header, message in zip(headers, messages)
                )
                chunks.append(text)
                remaining -= len(text) + (2 if len(chunks) > 1 else 0)
            if cursors and not chunks:
                raise ValueError("max_chars is too small for a timestamped conversation round")
            result = "\n\n".join(reversed(chunks))
            if cursors:
                await self.checkpoint_proactive(round_id, "conversation", result)
                async with self._sessions_lock:
                    current = await asyncio.to_thread(self._load_session_ids)
                    for entry in current:
                        if entry.session_id in cursors:
                            entry.last_read_message_id = cursors[entry.session_id]
                    await complete_in_thread(
                        self._write_json,
                        self.session_ids_path,
                        [entry.model_dump() for entry in current],
                    )
            return result

    @property
    def cursor(self) -> str:
        """Return the current replay watermark."""
        return f"{self._epoch}:{self._sequence}"

    async def _start(self) -> None:
        if self.app_context is None or not isinstance(self.agent_wrapper, CodexAgentWrapper):
            raise RuntimeError("PaRuntime requires an ApplicationContext and CodexAgentWrapper")
        await complete_in_thread(self._initialize_files)
        await self.reminders.window(await self.get_proactive_config())
        sessions = await asyncio.to_thread(self._load_session_ids)
        self.session_id = sessions[-1].session_id if sessions else None
        self._closed = False
        self.started_at = self._now()
        self._unsubscribe_activity = self.agent_wrapper.app_server.watch_activity(self.observe_task_activity)
        self._invalidate("cursor_expired")
        self.select_session()

    def _initialize_files(self) -> None:
        """Create missing PA files before jobs start, preserving all existing content."""
        renamed = {
            "tasks.json": self.cron_path,
            "proactive_inbox.txt": self.proactive_inbox_path,
            "reminders.jsonl": self.chat_inbox_path,
            "task_runs.jsonl": self.cron_runs_path,
            "proactive_requests.jsonl": self.inbox.receipts_path,
        }
        pending_layout = not self.chat_inbox_path.exists()
        legacy = [
            self._metadata_file(name) for name, target in renamed.items() if pending_layout or not target.exists()
        ]
        if pending_layout:
            legacy.append(self.tasks.checked(self.workspace_path / "reme_cron"))
        if any(path.exists() for path in legacy):
            raise ValueError(
                "Legacy PA layout requires offline migration: python -m scripts.migrate_pa_tasks WORKSPACE",
            )
        self._write_text(self.session_ids_path, "[]\n", create_only=True)
        self._load_session_ids()
        if not self.proactive_config_path.exists():
            if self.default_proactive is None:
                raise ValueError("PA requires default_proactive when proactive.json does not exist")
            self._write_text(
                self.proactive_config_path,
                self.default_proactive.model_dump_json(indent=2) + "\n",
                create_only=True,
            )
        self._load_proactive_config()
        self.tasks.initialize()
        self.inbox.initialize()
        self.temp_tasks.initialize()
        for path in (
            self.proactive_inbox_path,
            self.chat_inbox_path,
            self.cron_runs_path,
            self.proactive_context_path,
            self.proactive_todo_path,
        ):
            self._write_text(path, "", create_only=True)
        self.reminders.restore()

    async def announce_messages(self) -> None:
        """Offer an immediate result independently of the selected chat's generation lock."""
        if self._closed:
            return
        messages = await asyncio.to_thread(self.reminders.inbox_messages)
        if self._closed:
            return
        self._revision += 1
        status, flags = self._activity()
        self._publish(
            "status_changed",
            status=status,
            active_flags=flags,
            activities=self._activities,
            task_activities=self._task_activities,
            inbox_messages=messages,
        )

    def _write_json(self, path: Path, data) -> None:
        self._write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def _write_text(self, path: Path, content: str, *, create_only: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not path.is_file():
                raise ValueError(f"PA state must be a regular file: {path}")
            if create_only:
                return
            # Respect read-only state files instead of replacing them through directory permissions.
            with path.open("r+"):
                pass
        temporary = path.with_name(f".pa-{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if create_only:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass
            else:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @asynccontextmanager
    async def session_operation(self):
        """Reject overlapping chat operations and publish activity changes."""
        if self.session_lock.locked():
            raise RuntimeError("PA is busy; wait for the current operation to finish")
        try:
            async with self.session_lock:
                self.publish_status()
                yield
        finally:
            self.publish_status()

    @asynccontextmanager
    async def chat_session(self, **reply_kwargs):
        """Select or create the application-owned conversation for one request."""
        async with self.session_operation():
            session_id = self.session_id
            if session_id is None:
                session_id = await finish_cleanup(self._new_session(reply_kwargs))
            else:
                await self.remember_session(session_id)
            async with self.memory_lock:
                yield session_id

    async def new_session(self, **reply_kwargs) -> str:
        """Create and select an empty conversation without model generation."""
        async with self.session_operation():
            return await finish_cleanup(self._new_session(reply_kwargs))

    async def select_existing_session(self, session_id: str) -> str:
        """Select a readable source conversation without overlapping Main work."""
        async with self.session_operation():
            await self.agent_wrapper.read_session(session_id)

            async def commit():
                await self.remember_session(session_id)
                self.session_id = session_id
                self.select_session()

            await finish_cleanup(commit())
        return session_id

    async def _new_session(self, reply_kwargs: dict) -> str:
        session_id = await self.agent_wrapper.create_session(**reply_kwargs)
        previous = self.session_id
        try:
            await self.remember_session(session_id)
            self.session_id = session_id
            self.select_session()
        except BaseException:
            try:
                await self.agent_wrapper.release_session(session_id)
            except Exception as exc:
                self.logger.warning(f"Failed to release unselected PA session {session_id}: {exc}")
            raise
        if previous is not None:
            try:
                await self.agent_wrapper.release_session(previous)
            except Exception as exc:
                self.logger.warning(f"Failed to release previous PA session {previous}: {exc}")
        return session_id

    async def get_selected_session(self) -> tuple[dict, str | None]:
        """Report state without queueing behind chat generation."""
        while True:
            session_id = self.session_id
            status, error = None, None
            if self.session_lock.locked():
                status = "active"
            elif session_id is not None:
                try:
                    status = (await self.agent_wrapper.get_session_status(session_id)).status
                except Exception as exc:
                    status, error = "error", str(exc)
                if session_id != self.session_id:
                    continue
                if self.session_lock.locked():
                    status, error = "active", None
            return {"session_id": session_id, "status": status}, error

    def _wake(self) -> None:
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def _invalidate(self, reason: str) -> None:
        self._epoch = uuid4().hex
        self._sequence = 0
        self._revision += 1
        self._buffer.clear()
        self._buffer_bytes = 0
        self._fingerprints.clear()
        self._reset_reason = reason
        self._needs_refresh = True
        self._wake()

    def _publish(self, event_type: str, **values) -> None:
        self._sequence += 1
        event = PaEvent(
            type=event_type,
            cursor=self.cursor,
            session_id=self.session_id,
            notifications=self.notifications,
            **values,
        )
        serialized = event.model_dump_json(exclude_none=True)
        size = len(serialized.encode("utf-8"))
        self._buffer.append((self._sequence, serialized, size))
        self._buffer_bytes += size
        while len(self._buffer) > self.max_events or self._buffer_bytes > self.max_buffer_bytes:
            self._buffer_bytes -= self._buffer.popleft()[2]
        self._wake()

    def _activity(self) -> tuple[str | None, list[str]]:
        if self.session_lock.locked():
            return "active", self._native_status.active_flags if self._native_status is not None else []
        if self._native_status is not None:
            return self._native_status.status, self._native_status.active_flags
        return None, []

    def publish_status(self) -> None:
        """Merge foreground serialization with the native session's activity."""
        if self._closed:
            return
        status, flags = self._activity()
        signature = (
            self.session_id,
            status,
            tuple(flags),
            tuple((source, value.model_dump_json()) for source, value in sorted(self._activities.items())),
            tuple((run_id, value.model_dump_json()) for run_id, value in sorted(self._task_activities.items())),
            self.notifications.model_dump_json() if self.notifications is not None else None,
        )
        if signature != self._published_status:
            self._revision += 1
            self._published_status = signature
            self._publish(
                "status_changed",
                status=status,
                active_flags=flags,
                activities=self._activities,
                task_activities=self._task_activities,
            )

    def start_task_activity(self, run: PaTaskRun) -> None:
        """Keep a run busy through both model queries and Markdown publication."""
        if self._closed:
            return
        self._task_activities[run.run_id] = AgentActivity(
            session_id=run.agent_session_id or run.run_id,
            turn_id=run.run_id,
            source="task",
            phase="working",
            run_id=run.run_id,
            task_id=run.task_id,
        )
        self.publish_status()

    def observe_task_activity(self, change: AgentActivity | None) -> None:
        """Task runs outlive selected-chat changes and never enter its transcript."""
        if self._closed:
            return
        if change is None:
            self._task_activities.clear()
        elif change.source == "task" and change.run_id in self._task_activities:
            if change.phase == "completed":
                change = change.model_copy(update={"phase": "working", "tool": None})
            self._task_activities[change.run_id] = change
        else:
            return
        self.publish_status()

    def finish_task_activity(self, run_id: str) -> None:
        """Retire the live presentation even when durable final recording fails."""
        self._task_activities.pop(run_id, None)
        self.publish_status()

    def select_session(self) -> None:
        """Follow the committed selection, dropping listeners for superseded sessions."""
        if self._closed:
            return
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        session_id = self.session_id
        self._activities.clear()
        self._native_status = SessionStatus(session_id=session_id, status="idle") if session_id else None
        self._fingerprints.clear()
        self._revision += 1
        self._needs_refresh = False
        if session_id and self.agent_wrapper.app_server is not None:

            def receive(change):
                if not self._closed and self.session_id == session_id:
                    self.observe(change)

            self._unsubscribe = self.agent_wrapper.app_server.watch_session(session_id, receive)
        self._publish("session_changed")
        self.publish_status()

    @staticmethod
    def _fingerprint(message: SessionMessage) -> bytes:
        return sha256(message.model_dump_json().encode("utf-8")).digest()

    def observe(self, change: AgentActivity | SessionMessage | SessionSnapshot | SessionStatus | None) -> None:
        """Accept published history/status from the session owner without blocking it."""
        if self._closed:
            return
        self._revision += 1
        session_id = self.session_id
        if change is None:
            self._activities.clear()
            self._native_status = SessionStatus(session_id=session_id, status="error") if session_id else None
            self._invalidate("upstream_disconnected")
        elif isinstance(change, AgentActivity):
            if change.session_id != session_id or change.source == "task":
                return
            self._activities[change.source] = change
        elif isinstance(change, SessionMessage):
            fingerprint = self._fingerprint(change)
            if self._fingerprints.get(change.id) != fingerprint:
                self._fingerprints[change.id] = fingerprint
                self._publish("message_upsert", message=change)
        elif change.session_id == session_id:
            if isinstance(change, SessionSnapshot):
                self._native_status = change.state
                for message in change.messages:
                    self.observe(message)
                self._needs_refresh = False
            else:
                self._native_status = change
        self.publish_status()

    async def _snapshot(self) -> PaEvent | None:
        # Retry if a switch or owner notification crosses the history read.
        # There is no await between the final version check and its watermark.
        while not self._closed:
            revision, session_id = self._revision, self.session_id
            snapshot, error = None, None
            try:
                await self.reminders.window(await self.get_proactive_config())
            except (OSError, ValueError) as exc:
                self.notifications = None
                self.logger.warning(f"Ordinary notifications paused by invalid configuration: {exc}")
            try:
                if session_id is not None:
                    async with asyncio.timeout(30):
                        snapshot = await self.agent_wrapper.read_session(session_id)
            except Exception as exc:
                error = str(exc)
            inbox_messages = await asyncio.to_thread(self.reminders.inbox_messages)
            if self._closed:
                return None
            if revision != self._revision or session_id != self.session_id:
                continue
            self._needs_refresh = error is not None
            if snapshot is not None:
                self._native_status = snapshot.state
            elif session_id is not None:
                self._native_status = SessionStatus(session_id=session_id, status="error")
            self.publish_status()
            status, flags = self._activity()
            return PaEvent(
                type="snapshot",
                cursor=None if error is not None else self.cursor,
                session_id=session_id,
                status=status,
                active_flags=flags,
                activities=self._activities,
                task_activities=self._task_activities,
                notifications=self.notifications,
                inbox_messages=inbox_messages,
                snapshot=snapshot,
                error=error,
            )
        return None

    def _position(self, cursor: str | None) -> tuple[int, str | None]:
        match = _CURSOR.fullmatch(cursor) if cursor is not None else None
        if match is None:
            return 0, "invalid_cursor"
        epoch, sequence = match[1], int(match[2])
        oldest = self._buffer[0][0] if self._buffer else self._sequence + 1
        if epoch != self._epoch:
            return 0, self._reset_reason
        if sequence > self._sequence or sequence < oldest - 1:
            return 0, "cursor_expired"
        return sequence, None

    async def events(self, cursor: str | None = None) -> AsyncIterator[PaEvent]:
        """Replay changes, reset gaps to snapshots, and wait without polling healthy history."""
        if cursor is not None and (not isinstance(cursor, str) or len(cursor) > 128):
            raise ValueError("cursor must be a string of at most 128 characters")
        retry_snapshot = False
        while not self._closed:
            changed = self._changed
            if retry_snapshot:
                # No delta may advance a reader beyond history it has not recovered.
                try:
                    await asyncio.wait_for(changed.wait(), timeout=self.heartbeat_seconds)
                except asyncio.TimeoutError:
                    pass
                snapshot = await self._snapshot()
                if snapshot is None:
                    return
                retry_snapshot = snapshot.error is not None
                cursor = snapshot.cursor
                yield snapshot
                continue
            sequence, reason = self._position(cursor)
            if reason is not None:
                if cursor is not None:
                    yield PaEvent(type="reset", reason=reason)
                snapshot = await self._snapshot()
                if snapshot is None:
                    return
                retry_snapshot = snapshot.error is not None
                cursor = snapshot.cursor
                yield snapshot
                continue
            pending = next((data for seq, data, _size in self._buffer if seq > sequence), None)
            if pending is not None:
                event = PaEvent.model_validate_json(pending)
                cursor = event.cursor
                yield event
                continue
            try:
                await asyncio.wait_for(changed.wait(), timeout=self.heartbeat_seconds)
            except asyncio.TimeoutError:
                if self._needs_refresh:
                    snapshot = await self._snapshot()
                    if snapshot is None:
                        return
                    retry_snapshot = snapshot.error is not None
                    cursor = snapshot.cursor
                    yield snapshot
                else:
                    yield PaEvent(type="heartbeat", session_id=self.session_id)

    async def _close(self) -> None:
        self._closed = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._unsubscribe_activity is not None:
            self._unsubscribe_activity()
            self._unsubscribe_activity = None
        self._buffer.clear()
        self._buffer_bytes = 0
        self._fingerprints.clear()
        self._activities.clear()
        self._task_activities.clear()
        self._wake()
