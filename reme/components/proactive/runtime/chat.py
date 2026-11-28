"""Remy's foreground state and protected notification decisions."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from hashlib import sha256
from uuid import uuid4

from .base import BaseAgentRuntime
from ..inbox import NotificationInbox
from ..outbox import ChatOutbox
from ....constants import CHAT_AGENT_NAME, PROACTIVE_MESSAGE_ID_PREFIX
from ....core.registry import R
from ....schema.agent_outbox import ChatDelivery
from ....schema.proactive_runtime import ChatAgentConfig, NotificationConfig
from ....utils.async_utils import complete_in_thread, finish_cleanup
from ....utils.files.path_lock import get_path_lock


@dataclass
class NotificationTurn:
    """An invocation-scoped capability; it never survives the owning fork."""

    session_id: str
    priority: str
    token: str
    ids: tuple[str, ...]
    items: list[dict]
    versions: dict[str, str]
    chat_revision: int
    session_revision: str
    session_turn_id: str | None
    active_delivery: bool
    eligible: object = None
    id: str = field(default_factory=lambda: uuid4().hex)
    content: str | None = None
    done: bool = False
    delivery_id: str | None = None
    message_id: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@R.register("chat")
class ChatAgentRuntime(BaseAgentRuntime):
    """Own Remy resources without depending back on AgentManager."""

    def __init__(self, initial_memory="", notification_config=None, enabled=True, **kwargs):
        super().__init__(**kwargs)
        if self.name != CHAT_AGENT_NAME:
            raise ValueError(f"The Chat Runtime must be named {CHAT_AGENT_NAME}")
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        self.enabled = enabled
        self.initial_memory = ChatAgentConfig(initial_memory=initial_memory).initial_memory
        self.notification_config = NotificationConfig.model_validate(notification_config or {})
        self.chat_revision = 0
        self.last_activity_at = self.now()
        self.foreground_tasks = set()
        self.foreground_task = None
        self.notification_worker = None
        self.active_turn = None
        self.retry_at = None
        self.failures = 0
        self.blocked = False

    @property
    def memory_path(self):
        """Locate Remy's user-owned personal memory document."""
        return self.files.checked(self.directory / "MEMORY.md")

    @property
    def current_tasks(self):
        """Include every owned active worker and queued invocation."""
        return super().current_tasks | ({self.notification_worker} if self.notification_worker is not None else set())

    async def _start(self):
        self.inbox = NotificationInbox(
            name=f"{self.name}_inbox",
            directory=self.to_workspace_relative(self.directory / "inbox"),
            app_context=self.app_context,
        )
        self.outbox = ChatOutbox(
            name=f"{self.name}_outbox",
            directory=self.to_workspace_relative(self.directory / "outbox/chat"),
            wrapper=self.agent_wrapper,
            app_context=self.app_context,
        )
        await self.start_children()
        await complete_in_thread(self.files.write, self.memory_path, self.initial_memory, create_only=True)

    @asynccontextmanager
    async def chat_turn(self):
        """Serialize foreground work and invalidate older notification decisions."""
        self.chat_revision += 1
        task = asyncio.current_task()
        self.foreground_tasks.add(task)
        try:
            async with self.execution():
                if not self.enabled:
                    raise RuntimeError("Chat Runtime is disabled")
                self.foreground_task = task
                try:
                    yield
                finally:
                    self.foreground_task = None
        finally:
            self.foreground_tasks.discard(task)
            self.last_activity_at = self.now()
            self.live_turns.notify()

    @property
    def busy(self):
        """Include queued foreground requests in the idle decision."""
        return bool(self.foreground_tasks) or self.execution_lock.locked()

    @property
    def foreground_waiting(self):
        """Queued user inputs take precedence over an automatic notification."""
        return bool(self.foreground_tasks - {self.foreground_task})

    def record_version(self, path):
        """Hash the exact source content used by a notification decision."""
        return sha256(self.files.read(self.files.path(path)).encode()).hexdigest()

    async def reminder_item(self, reference):
        """Resolve a reference into the current canonical reminder view."""
        return (await asyncio.to_thread(self._reminder_material, reference))[0]

    def _reminder_material(self, reference):
        reminder, version = self.inbox.material(reference)
        if reminder is None:
            raise ValueError("Source record no longer requests a reminder")
        return {
            **reminder.model_dump(mode="json"),
            "from_agent_name": reference.source_agent,
            "delivered_at": reference.created_at.isoformat(),
            "notification_id": reference.id,
        }, version

    @asynccontextmanager
    async def notification_turn(self, session_id, priority, identifiers, *, format_item, eligible=None):
        """Own a selected batch capability until the decision fork exits."""
        if self.active_turn is not None:
            raise RuntimeError("A notification fork is already running")
        revision = self.chat_revision
        if self.foreground_waiting:
            raise RuntimeError("A foreground request is waiting; reassess notifications")
        batch = await self.inbox.claim(item_ids=list(identifiers))
        if batch is None:
            raise ValueError("No pending notifications in the selected batch")
        turn = None
        try:
            items, versions, ids = [], {}, []
            for raw in batch["items"]:
                reference = await self.inbox.reference(raw["id"])
                item, version = await asyncio.to_thread(self._reminder_material, reference)
                if item["priority"] != priority:
                    raise RuntimeError("Notification priority changed; reassess this batch")
                items.append(item)
                ids.append(reference.id)
                versions[reference.record_path] = version
            snapshot = await self.agent_wrapper.read_session(session_id)
            if self.chat_revision != revision or self.foreground_waiting:
                raise RuntimeError("Chat conversation changed; reassess notifications")
            if len("\n\n---\n\n".join(format_item(item) for item in items).encode()) > self.inbox.claim_max_bytes:
                raise ValueError("Notification content exceeds the batch byte budget")
            turn = NotificationTurn(
                session_id=session_id,
                priority=priority,
                token=batch["token"],
                ids=tuple(ids),
                items=items,
                versions=versions,
                chat_revision=revision,
                session_revision=snapshot.revision,
                session_turn_id=snapshot.last_turn_id,
                active_delivery=priority == "P0" and snapshot.state.status == "active",
                eligible=eligible,
            )
            self.active_turn = turn
            yield turn
        finally:
            if self.active_turn is turn:
                self.active_turn = None
            await finish_cleanup(self.inbox.nack(batch["token"]))

    def current_turn(self, identifier):
        """Require a live capability issued to the current decision fork."""
        if self.closed or self.active_turn is None or self.active_turn.id != identifier:
            raise ValueError("Chat reply requires an active notification fork")
        return self.active_turn

    async def _validate_turn(self, turn):
        self.current_turn(turn.id)
        current = await self.current_session()
        if current is None or current.session_id != turn.session_id or self.chat_revision != turn.chat_revision:
            raise RuntimeError("Chat conversation changed; reassess notifications")
        if not set(turn.ids) <= await self.inbox.pending_ids():
            raise RuntimeError("Notification batch is no longer pending")
        for path, version in turn.versions.items():
            if await asyncio.to_thread(self.record_version, path) != version:
                raise RuntimeError("Notification source changed; reassess notifications")
        if turn.eligible is not None and not await turn.eligible(turn.items):
            raise RuntimeError("Notifications are no longer eligible")
        if self.chat_revision != turn.chat_revision or self.foreground_waiting:
            raise RuntimeError("Chat conversation changed; reassess notifications")

    async def submit_reply(self, turn_id, action, content=None):
        """Commit one explicit send or skip decision after revalidation."""
        turn = self.current_turn(turn_id)
        text = content.strip() if action == "send" else ""
        async with turn.lock:
            self.current_turn(turn_id)
            if turn.content is not None and turn.content != text:
                raise ValueError("Notification decision has already been submitted")
            if turn.done:
                return {
                    "sent": bool(turn.content),
                    "message_id": PROACTIVE_MESSAGE_ID_PREFIX + turn.message_id if turn.message_id else None,
                }
            turn.content = text
            async with nullcontext() if turn.active_delivery else self.execution_lock:
                async with AsyncExitStack() as stack:
                    for path in sorted(turn.versions):
                        await stack.enter_async_context(await get_path_lock(self.files.path(path)))
                    await self._validate_turn(turn)
                    if action == "skip":
                        await stack.enter_async_context(
                            self.agent_wrapper.session_guard(turn.session_id, expected_revision=turn.session_revision),
                        )
                        await self._validate_turn(turn)
                        await finish_cleanup(self.inbox.ack(turn.token, metadata={"decision": "skip"}))
                    else:
                        key = "\0".join((turn.session_id, *sorted(turn.ids)))
                        identifier = "chat-" + turn.id
                        turn.delivery_id = identifier
                        turn.message_id = "reminder-" + sha256(key.encode()).hexdigest()
                        delivery = await self.outbox.read(identifier)
                        if delivery is None:
                            delivery = await self.outbox.prepare(
                                ChatDelivery(
                                    id=identifier,
                                    notification_ids=list(turn.ids),
                                    record_versions=turn.versions,
                                    session_id=turn.session_id,
                                    chat_revision=turn.chat_revision,
                                    session_revision=turn.session_revision,
                                    session_turn_id=turn.session_turn_id if turn.active_delivery else None,
                                    content=text,
                                    message_id=turn.message_id,
                                    created_at=self.now(),
                                ),
                            )

                        async def commit():
                            await self.outbox.deliver(delivery.id, validate=lambda: self._validate_turn(turn))
                            await self.inbox.ack(
                                turn.token,
                                metadata={"decision": "send", "chat_delivery_id": delivery.id},
                            )
                            await self.outbox.acknowledge(delivery.id, self.now())

                        await finish_cleanup(commit())
                    turn.done = True
            self.live_turns.notify()
            return {
                "sent": bool(text),
                "message_id": PROACTIVE_MESSAGE_ID_PREFIX + turn.message_id if turn.message_id else None,
            }

    async def recover_deliveries(self):
        """Resolve native side effects before starting another notification decision."""
        for delivery in await self.outbox.pending():
            try:
                receipts = [await self.inbox.processed(identifier) for identifier in delivery.notification_ids]
                if all(receipts):
                    if delivery.status == "prepared":
                        if await self.outbox.find_message(delivery) is None:
                            await self.outbox.abort(delivery.id, "Inputs were processed by another decision")
                            continue
                        delivery = await self.outbox.deliver(delivery.id)
                    if delivery.status == "sent":
                        await self.outbox.acknowledge(delivery.id, self.now())
                    continue
                message = await self.outbox.find_message(delivery)
                if message is None:
                    if delivery.status == "sent":
                        raise RuntimeError("Sent Chat reply is unavailable in native history; recovery is pending")
                    await self.outbox.abort(delivery.id, "Interrupted before native delivery; reassess notifications")
                    continue
                if message.content != delivery.content:
                    raise ValueError("Native Chat reply conflicts with its saved intention")
                await self.outbox.deliver(delivery.id)
                pending = await self.inbox.pending_ids()
                identifiers = [identifier for identifier in delivery.notification_ids if identifier in pending]
                if identifiers:
                    batch = await self.inbox.claim(item_ids=identifiers)
                    if batch is None:
                        continue
                    try:
                        await self.inbox.ack(
                            batch["token"],
                            metadata={"decision": "send", "chat_delivery_id": delivery.id},
                        )
                    finally:
                        await self.inbox.nack(batch["token"])
                await self.outbox.acknowledge(delivery.id, self.now())
                self.error = None
            except (OSError, ValueError, RuntimeError) as exc:
                self.error = str(exc)
        return {identifier for delivery in await self.outbox.pending() for identifier in delivery.notification_ids}
