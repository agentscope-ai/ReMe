"""Chat reminder scheduling and disposable-fork decisions over daily records."""

import asyncio
import json
from collections import defaultdict
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from hashlib import sha256
from uuid import uuid4

from ....schema.response import JOB_ERROR_METADATA_KEY
from ....constants import CHAT_AGENT_NAME, PROACTIVE_MESSAGE_ID_PREFIX
from ....schema.agent_runtime import NotificationConfig
from ....schema.agent_session import SessionMessage
from ....utils.async_utils import finish_cleanup


@dataclass
class InboxTurn:
    """One disposable fork's tools and explicit delivery decision."""

    session_id: str
    priority: str
    msg_ids: tuple[str, ...]
    chat_revision: int = 0
    session_revision: str | None = None
    session_turn_id: str | None = None
    active_delivery: bool = False
    id: str = field(default_factory=lambda: uuid4().hex)
    items: list[dict] | None = None
    content: str | None = None
    message: SessionMessage | None = None
    done: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def message_id(self):
        """Keep retries of the same batch idempotent."""
        value = "\0".join((self.session_id, *sorted(self.msg_ids)))
        return "reminder-" + sha256(value.encode()).hexdigest()


class ChatNotifications:
    """Coordinate Chat wakeups while daily records hold reminders and acknowledgements."""

    def __init__(self, runtime, config: NotificationConfig):
        self.runtime = runtime
        self.config = config
        self.lock = asyncio.Lock()
        self.last_activity_at = self.runtime.now()
        self.worker = None
        self.retry_at = None
        self.active_turn: InboxTurn | None = None
        self.chat_revision = 0
        self.error = None
        self.failures = 0
        self.blocked = False
        self.last_started_at = None
        self.last_completed_at = None
        self._reported_oversized: set[str] = set()

    def reset(self):
        """Restart idle and retry timers after owned work has stopped."""
        self.last_activity_at = self.runtime.now()
        self.retry_at = self.error = None
        self.failures = 0
        self.blocked = False

    @property
    def pending(self):
        """Project unread reminders without duplicating queue state."""
        return {ref for ref, item in self.runtime.outbox.records.items() if item.read_at is None}

    @property
    def read_refs(self):
        """Keep the same bounded reference horizon as the inbox component."""
        records = self.runtime.outbox.records
        refs = sorted(
            (ref for ref, item in records.items() if item.read_at is not None),
            key=lambda ref: (records[ref].read_at, ref),
        )[-self.runtime.inbox.max_read_items :]
        return set(refs)

    def activity(self):
        """Restart idle timing after every foreground or internally requested Chat turn."""
        self.last_activity_at = self.runtime.now()

    def foreground(self):
        """Invalidate an older reminder decision as soon as a user turn is queued."""
        self.chat_revision += 1

    async def read(self, msg_ids=None):
        """Expand pending references without duplicating canonical message state."""
        async with self.lock:
            outbox = self.runtime.outbox
            refs = self.pending if msg_ids is None else self.pending.intersection(msg_ids)
            return {"items": [outbox.message(ref) for ref in sorted(refs, key=outbox.order)]}

    async def consume(self, msg_ids):
        """Acknowledge the messages covered by a committed decision."""
        async with self.lock:
            await self._consume(msg_ids, self.runtime.outbox.update)

    async def _consume(self, msg_ids, update):
        identifiers = dict.fromkeys(msg_ids)
        if identifiers.keys() - (self.read_refs | self.pending):
            raise ValueError("unknown inbox message ID")

        async def save():
            for identifier in identifiers:
                await update(identifier, read_at=self.runtime.now())

        await finish_cleanup(save())

    def _bounded(self, refs):
        """Bound one inference batch using the configured inbox count and byte budgets."""
        inbox = self.runtime.inbox
        limit = min(20, inbox.claim_max_items, inbox.max_read_items, inbox.max_pending_items)
        byte_limit = remaining = min(inbox.claim_max_bytes, inbox.max_pending_bytes)
        selected = []
        for ref in refs:
            if len(selected) >= limit:
                break
            size = (
                len(
                    json.dumps(self.runtime.outbox.event(ref), ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8",
                    ),
                )
                + 1
            )
            if size > byte_limit:
                if ref not in self._reported_oversized:
                    self._reported_oversized.add(ref)
                    self.runtime.logger.warning(
                        f"Chat reminder {ref} remains pending: "
                        f"reference is {size} bytes, exceeding the {byte_limit}-byte batch limit",
                    )
                continue
            if size > remaining:
                break
            selected.append(ref)
            remaining -= size
        return tuple(selected)

    @asynccontextmanager
    async def turn(self, session_id, priority, msg_ids):
        """Own one in-memory batch; canonical acknowledgements survive cancellation and restart."""
        async with self.lock:
            if self.active_turn is not None:
                raise RuntimeError("an inbox fork is already running")
            pending = self.pending
            refs = [
                identifier
                for identifier in dict.fromkeys(msg_ids)
                if identifier in pending and self.runtime.outbox.records[identifier].priority == priority
            ]
            selected = self._bounded(refs)
            if not selected:
                raise ValueError("no pending messages at the selected priority")
            turn = InboxTurn(session_id, priority, selected, chat_revision=self.chat_revision)
            self.active_turn = turn
        try:
            snapshot = await self.runtime.agent_wrapper.read_session(session_id)
            turn.session_revision = snapshot.revision
            turn.session_turn_id = snapshot.last_turn_id
            turn.active_delivery = priority == "P0" and snapshot.state.status == "active"
            messages = {message.id: message for message in snapshot.messages if message.role == "assistant"}
            recovered = {}
            for identifier in selected:
                item = self.runtime.outbox.records[identifier]
                for message_id in item.chat_message_ids:
                    if message_id in messages:
                        recovered[identifier] = messages[message_id]
                        break
            if recovered:
                # Native acceptance is the durable decision; finish only its source acknowledgements.
                await finish_cleanup(self.consume(list(recovered)))
                remaining = tuple(identifier for identifier in selected if identifier not in recovered)
                if remaining:
                    turn.msg_ids = remaining
                else:
                    turn.message = recovered[selected[0]]
                    turn.content, turn.done = turn.message.content, True
            yield turn
        finally:
            self.active_turn = None

    def current_turn(self, identifier):
        """Reject foreground calls and capabilities left over from a finished fork."""
        if self.active_turn is None or identifier != self.active_turn.id:
            raise ValueError("inbox tools require an active reminder fork")
        return self.active_turn

    async def read_turn(self, turn):
        """Expose only this wakeup's batch; later arrivals wait for another fork."""
        async with turn.lock:
            self.current_turn(turn.id)
            if turn.items is None:
                turn.items = (await self.read(turn.msg_ids))["items"]
            return {"items": turn.items}

    async def send(self, turn, content):
        """Commit the explicit text or silence, then consume only what the fork read."""
        content = content.strip()
        async with turn.lock:
            self.current_turn(turn.id)
            if turn.items is None:
                raise ValueError("read_chat_reminders must be called before submit_chat_reminder")
            if content and not turn.items:
                raise ValueError("no reminders were read")
            if turn.content is not None and content != turn.content:
                raise ValueError("the reminder decision has already been submitted")
            turn.content = content
            if not turn.done:
                async with nullcontext() if turn.active_delivery else self.runtime.chat_lock:
                    current = await self.runtime.sessions.current(CHAT_AGENT_NAME)
                    if self.runtime.closed or current is None or current.session_id != turn.session_id:
                        raise RuntimeError("Chat session changed; retry the reminders in the current conversation")
                    if self.chat_revision != turn.chat_revision:
                        raise RuntimeError("Chat conversation changed; reassess reminders after the latest user turn")
                    await self.runtime.outbox.refresh()

                    async def commit(update):
                        if content and turn.message is None:
                            message_id = turn.message_id
                            for item in turn.items:
                                await update(
                                    item["id"],
                                    chat_message_ids=[f"{PROACTIVE_MESSAGE_ID_PREFIX}{message_id}"],
                                )
                            turn.message = await self.runtime.agent_wrapper.append_session(
                                turn.session_id,
                                content,
                                message_id=message_id,
                                role="assistant",
                                expected_turn_id=(turn.session_turn_id if turn.active_delivery else None),
                                expected_revision=(None if turn.active_delivery else turn.session_revision),
                                delivery=("immediate" if turn.priority == "P0" else "idle"),
                            )
                        current = await self.runtime.sessions.current(CHAT_AGENT_NAME)
                        if (
                            current is None
                            or current.session_id != turn.session_id
                            or self.chat_revision != turn.chat_revision
                        ):
                            raise RuntimeError("Chat conversation changed during delivery; reassess reminders")
                        await self._consume([item["id"] for item in turn.items], update)
                        turn.done = True

                    guard = (
                        self.runtime.agent_wrapper.session_guard(
                            turn.session_id,
                            expected_revision=turn.session_revision,
                        )
                        if not content
                        else nullcontext()
                    )
                    async with (
                        guard,
                        self.lock,
                        self.runtime.outbox.decision(turn.items) as update,
                    ):
                        agents = await asyncio.to_thread(self.runtime.agents.load)
                        pending = self.pending
                        if not all(
                            item["id"] in pending and self.runtime.outbox.allows(item["id"], agents, self.runtime.now())
                            for item in turn.items
                        ):
                            raise RuntimeError("reminders are no longer eligible for delivery")
                        await finish_cleanup(commit(update))
            return {
                "sent": bool(content),
                "message_id": turn.message.id if turn.message else None,
            }

    async def tick(self, run_job):
        """Coalesce eligible messages into one lifecycle-owned Chat wakeup."""
        if self.worker is not None or self.runtime.closed:
            return
        await self.runtime.outbox.sync()
        if self.blocked:
            return
        now = self.runtime.now()
        if self.retry_at is not None and now < self.retry_at:
            return
        agents = await asyncio.to_thread(self.runtime.agents.load)
        async with self.lock:
            if self.worker is not None or self.runtime.closed:
                return
            idle = max(0, now.timestamp() - self.last_activity_at.timestamp())
            busy = bool(self.runtime.chat_tasks) or self.runtime.chat_lock.locked()
            pending = self.pending
            self._reported_oversized.intersection_update(pending)
            candidates = defaultdict(list)
            for ref in pending:
                item = self.runtime.outbox.records[ref]
                if not self.runtime.outbox.allows(ref, agents, now):
                    continue
                if item.priority == "P0":
                    if (now - item.delivered_at).total_seconds() < self.config.p0_delay_seconds:
                        continue
                elif busy or idle < getattr(self.config, f"{item.priority.lower()}_idle_seconds"):
                    continue
                candidates[item.priority].append(ref)
            for priority, refs in sorted(candidates.items()):
                refs.sort(key=self.runtime.outbox.order)
                identifiers = self._bounded(refs)
                if not identifiers:
                    continue
                self.worker = asyncio.create_task(
                    self._wake(run_job, priority, identifiers),
                    name="inbox:chat",
                )
                break

    async def _wake(self, run_job, priority, identifiers):
        permanent = False
        self.last_started_at = self.runtime.now()
        try:
            response = await run_job("process_chat_reminders", priority=priority, msg_ids=identifiers)
            if not response.success:
                failure = response.metadata.get(JOB_ERROR_METADATA_KEY, {})
                permanent = failure.get("type") in {"TypeError", "ImportError", "ModuleNotFoundError"}
                raise RuntimeError(str(response.answer))
            if isinstance(response.answer, dict) and response.answer.get("skipped"):
                if response.answer["skipped"] == "busy":
                    self.retry_at = self.runtime.now() + timedelta(seconds=self.runtime.retry_seconds)
                return
            self.retry_at = self.error = None
            self.failures = 0
            self.last_completed_at = self.runtime.now()
        except Exception as exc:
            self.error = (str(exc).strip() or type(exc).__name__)[:2000]
            self.failures += 1
            self.blocked = permanent or isinstance(exc, (TypeError, ImportError))
            delay = min(300, max(1, self.runtime.retry_seconds) * 2 ** min(self.failures - 1, 8))
            self.retry_at = None if self.blocked else self.runtime.now() + timedelta(seconds=delay)
            recovery = "repair the environment and restart" if self.blocked else f"retry in {delay:g}s"
            self.runtime.logger.error(f"Inbox wakeup failed ({recovery}): {self.error}")
        finally:
            self.worker = None

    async def close(self):
        """Cancel the owned wakeup before closing shared agent infrastructure."""
        if self.worker is not None:
            task = self.worker
            task.cancel()
            await finish_cleanup(asyncio.gather(task, return_exceptions=True))
            self.worker = None
