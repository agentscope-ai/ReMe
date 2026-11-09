"""Application-owned, bounded notifications for the selected PA conversation."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import threading
from uuid import uuid4
from zoneinfo import ZoneInfo

from ..agent_wrapper import CodexAgentWrapper
from ..base_component import BaseComponent
from ..component_registry import R
from .hints import ProactiveHints
from .reminders import ReminderDelivery, message_content
from .tasks import TaskFiles
from .inbox import ProactiveInbox
from .temp_tasks import TempTasks
from .files import PaFiles
from .sessions import PaSessions
from .events import PaEventStream
from .views import PaViews
from ...schema.pa_proactive import PaNotificationWindow, PaProactiveConfig
from ...schema.pa_hint import HintPeriod
from ...schema.reminder import ReminderKind
from ...schema.pa_task import PaTask
from ...utils.async_utils import complete_in_thread, finish_cleanup
from ...constants import DEFAULT_MAX_FILE_BYTES


@dataclass(slots=True)
class ProactiveRound:
    """One live agent round; deferred reminders outlive this object."""

    session_id: str | None
    id: str = field(default_factory=lambda: uuid4().hex)
    allow_message: bool = False
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
        self.files = PaFiles(self)
        self.views = PaViews(self)
        self.sessions = PaSessions(self)
        self.event_stream = PaEventStream(self, max_events, max_buffer_bytes, heartbeat_seconds)

    @property
    def proactive_inbox_path(self) -> Path:
        """Keep proactive feedback in the PA component's metadata directory."""
        return self.files.metadata_path("proactive_inbox.md")

    def now(self) -> datetime:
        """Return application-local time for scheduling and persisted timestamps."""
        timezone = self.app_context.app_config.timezone if self.app_context else None
        return datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()

    @property
    def proactive_config_path(self) -> Path:
        """Return the user-owned wakeup configuration path."""
        return self.files.metadata_path("proactive.json")

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
            self.event_stream.publish_status()
            raise

    async def proactive_skip_reason(self, config: PaProactiveConfig | None = None) -> str | None:
        """Apply local scheduling rules before spending any model tokens."""
        config = config or await self.get_proactive_config()
        window = await self.reminders.window(config)
        if not config.enabled:
            return "disabled"
        if not config.is_active(self.now()):
            return "outside_active_hours"
        return "day_closed" if window.closing_message_id is not None else None

    @property
    def proactive_context_path(self) -> Path:
        """Consumed inputs and partial progress, retained until the round completes."""
        return self.files.metadata_path("proactive_context.jsonl")

    def _checkpoint_context(self, round_id: str | None, kind: str, value) -> None:
        if round_id is not None and value:
            with self._work_context_lock:
                self.files.append_jsonl(
                    self.proactive_context_path,
                    {
                        "round_id": round_id,
                        "time": self.now().isoformat(),
                        "kind": kind,
                        "value": value,
                    },
                )

    async def checkpoint_proactive(self, round_id: str | None, kind: str, value) -> None:
        """Durably retain consumed inputs or progress for interrupted rounds."""
        await complete_in_thread(self._checkpoint_context, round_id, kind, value)

    async def clear_proactive_context(self) -> None:
        """Clear recovery records after the work query completes."""

        def clear():
            with self._work_context_lock:
                self.files.write_text(self.proactive_context_path, "")

        await complete_in_thread(clear)

    async def recover_proactive_context(self) -> str:
        """Replay consumed inputs until a work query has successfully handled them."""

        def read():
            with self._work_context_lock:
                if not self.proactive_context_path.exists():
                    return ""
                records = [json.loads(line) for line in self.files.read(self.proactive_context_path).splitlines()]
                inputs = [
                    record for record in records if record["kind"] in {"conversation", "feedback", "task_results"}
                ]
                return json.dumps(inputs, ensure_ascii=False) if inputs else ""

        return await asyncio.to_thread(read)

    async def write_proactive_inbox(self, message: str) -> dict:
        """Queue a message for the proactive agent's next wakeup."""
        return await finish_cleanup(self.inbox.write(message))

    async def read_proactive_inbox(
        self,
        max_bytes: int = DEFAULT_MAX_FILE_BYTES,
        *,
        round_id: str | None = None,
    ) -> str:
        """Return newly appended or edited Markdown, checkpointing it without deleting the user's inbox."""
        return await self.inbox.read(round_id=round_id, max_bytes=max_bytes)

    @property
    def chat_inbox_path(self) -> Path:
        """Append-only background messages and delivery states addressed to Main."""
        return self.files.metadata_path("chat_inbox.jsonl")

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
            if not current.allow_message:
                raise ValueError("send_chat_message requires an active proactive query")
            content = message_content(content, allow_empty=True)
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
        cron_task_id: str | None = None,
    ) -> list[dict]:
        """Read history and bind its version only to the current live execution."""
        current = self.tasks.execution(cron_task_id) if cron_task_id is not None else None
        records, version = await self.reminders.history(limit, offset)
        if round_id is not None:
            current = self.current_round
            if current is None or current.id != round_id:
                raise ValueError("History requires a live proactive round")
        elif cron_task_id is not None:
            self.tasks.execution(cron_task_id, current.state)
        if current is not None:
            if offset and current.history_version != version:
                raise ValueError("Reminder history changed; call get_chat_messages from offset=0 again")
            current.history_version = version
        return records

    @property
    def cron_path(self) -> Path:
        """User-editable cron definitions."""
        return self.files.metadata_path("cron.json")

    @property
    def cron_runs_path(self) -> Path:
        """Append-only cron start and finish records."""
        return self.files.metadata_path("cron_runs.jsonl")

    @property
    def session_ids_path(self) -> Path:
        """Recent chat sessions, oldest first, with persisted read cursors."""
        return self.files.metadata_path("session_ids.json")

    async def _start(self) -> None:
        if self.app_context is None or not isinstance(self.agent_wrapper, CodexAgentWrapper):
            raise RuntimeError("PaRuntime requires an ApplicationContext and CodexAgentWrapper")
        await complete_in_thread(self._initialize_files)
        await self.reminders.window(await self.get_proactive_config())
        sessions = await asyncio.to_thread(self.sessions.load)
        self.session_id = sessions[-1].session_id if sessions else None
        self.started_at = self.now()
        self.event_stream.start()

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
            self.files.metadata_path(name) for name, target in renamed.items() if pending_layout or not target.exists()
        ]
        if pending_layout:
            legacy.append(self.files.checked(self.workspace_path / "reme_cron"))
        if any(path.exists() for path in legacy):
            raise ValueError(
                "Legacy PA layout requires offline migration: python -m scripts.migrate_pa_tasks WORKSPACE",
            )
        self.files.write_text(self.session_ids_path, "[]\n", create_only=True)
        self.sessions.load()
        if not self.proactive_config_path.exists():
            if self.default_proactive is None:
                raise ValueError("PA requires default_proactive when proactive.json does not exist")
            self.files.write_text(
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
        ):
            self.files.write_text(path, "", create_only=True)
        self.reminders.restore()

    async def _close(self) -> None:
        await self.event_stream.close()
