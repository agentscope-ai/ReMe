"""Chat reminder scheduling and disposable-fork decisions over the shared Inbox component."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from hashlib import sha256
from uuid import uuid4

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
    """Coordinate Chat wakeups while reminder files retain canonical content and acknowledgements."""

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

    def reset(self):
        """Restart idle and retry timers after owned work has stopped."""
        self.last_activity_at = self.runtime.now()
        self.retry_at = self.error = None
        self.failures = 0
        self.blocked = False

    @property
    def pending(self):
        """Project delivered, unprocessed canonical reminders without duplicating queue state."""
        outbox = self.runtime.outbox
        return {
            ref.id: ref
            for ref in outbox.committed
            if outbox.records[ref].delivered_at is not None and outbox.records[ref].read_at is None
        }

    @property
    def read_refs(self):
        """Keep the same bounded reference horizon as the inbox component."""
        outbox = self.runtime.outbox
        refs = sorted(
            (ref for ref in outbox.committed if outbox.records[ref].read_at is not None),
            key=lambda ref: (outbox.records[ref].read_at, ref.id),
        )[-self.runtime.inbox.max_read_items :]
        return {ref.id: ref for ref in refs}

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
            refs = [ref for ref in self.pending.values() if msg_ids is None or ref.id in msg_ids]
            refs.sort(key=lambda ref: (outbox.records[ref].priority, outbox.records[ref].created_at, ref.id))
            return {
                "items": [
                    {
                        **outbox.records[ref].model_dump(mode="json"),
                        "from_agent_name": outbox.owners[ref],
                        "reminder_path": ref.file,
                    }
                    for ref in refs
                ],
            }

    async def consume(self, msg_ids):
        """Automatically acknowledge the messages covered by a committed decision."""
        async with self.lock:
            identifiers = list(dict.fromkeys(msg_ids))
            refs = {**self.read_refs, **self.pending}
            if any(identifier not in refs for identifier in identifiers):
                raise ValueError("unknown inbox message ID")

            async def save():
                for identifier in identifiers:
                    await self.runtime.outbox.update(refs[identifier], read_at=self.runtime.now())
                await self.runtime.outbox.reconcile()

            await finish_cleanup(save())

    @asynccontextmanager
    async def turn(self, session_id, priority, msg_ids):
        """Revoke the fork's tool scope on completion, failure and cancellation."""
        if self.active_turn is not None:
            raise RuntimeError("an inbox fork is already running")
        async with self.lock:
            pending = self.pending
            selected = tuple(
                identifier
                for identifier in dict.fromkeys(msg_ids)
                if identifier in pending and self.runtime.outbox.records[pending[identifier]].priority == priority
            )
        if not selected:
            raise ValueError("no pending messages at the selected priority")
        batch = await self.runtime.inbox.claim("Remy", item_ids=list(selected))
        if batch is None:
            raise RuntimeError("Chat inbox is already being processed")
        try:
            selected = tuple(item["id"] for item in batch["items"])
            snapshot = await self.runtime.agent_wrapper.read_session(session_id)
            turn = InboxTurn(
                session_id,
                priority,
                selected,
                chat_revision=self.chat_revision,
                session_revision=snapshot.revision,
                session_turn_id=snapshot.last_turn_id,
                active_delivery=priority == "P0" and snapshot.state.status == "active",
            )
            self.active_turn = turn
            yield turn
        finally:
            self.active_turn = None
            try:
                await finish_cleanup(self.runtime.inbox.nack("Remy", batch["token"]))
            except ValueError:
                pass

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
                raise ValueError("read_inbox must be called before send_chat_msg")
            if content and not turn.items:
                raise ValueError("no reminders were read")
            if turn.content is not None and content != turn.content:
                raise ValueError("the reminder decision has already been submitted")
            turn.content = content
            if not turn.done:
                async with nullcontext() if turn.active_delivery else self.runtime.chat_lock:
                    current = await self.runtime.sessions.current("Remy")
                    if self.runtime.closed or current is None or current.session_id != turn.session_id:
                        raise RuntimeError("Chat session changed; retry the reminders in the current conversation")
                    if self.chat_revision != turn.chat_revision:
                        raise RuntimeError("Chat conversation changed; reassess reminders after the latest user turn")
                    if not content:
                        snapshot = await self.runtime.agent_wrapper.read_session(turn.session_id)
                        if snapshot.revision != turn.session_revision:
                            raise RuntimeError("Chat conversation changed; reassess reminders before consuming them")
                    await self.runtime.outbox.refresh()
                    for item in turn.items:
                        ref = self.pending.get(item["id"])
                        latest = self.runtime.outbox.records.get(ref)
                        if (
                            latest is None
                            or latest.read_at is not None
                            or any(getattr(latest, key) != item[key] for key in ("priority", "message", "paths"))
                        ):
                            raise RuntimeError("reminders changed; reassess the current inbox before submitting")
                    if content:
                        agents = await asyncio.to_thread(self.runtime.agents.load)
                        if not all(
                            item["id"] in self.pending
                            and self._allows(self.pending[item["id"]], agents, self.runtime.now())
                            for item in turn.items
                        ):
                            raise RuntimeError("reminders are no longer eligible for delivery")

                    async def commit():
                        if content:
                            for item in turn.items:
                                await self.runtime.outbox.update(
                                    self.pending[item["id"]],
                                    chat_message_ids=[f"reme-proactive-{turn.message_id}"],
                                )
                        if content and turn.message is None:
                            turn.message = await self.runtime.agent_wrapper.append_session(
                                turn.session_id,
                                content,
                                message_id=turn.message_id,
                                role="assistant",
                                expected_turn_id=turn.session_turn_id if turn.active_delivery else None,
                                expected_revision=None if turn.active_delivery else turn.session_revision,
                                delivery="immediate" if turn.priority == "P0" else "idle",
                            )
                        current = await self.runtime.sessions.current("Remy")
                        if (
                            current is None
                            or current.session_id != turn.session_id
                            or self.chat_revision != turn.chat_revision
                        ):
                            raise RuntimeError("Chat conversation changed during delivery; reassess reminders")
                        await self.consume([item["id"] for item in turn.items])
                        turn.done = True

                    await finish_cleanup(commit())
            return {"sent": bool(content), "message_id": turn.message.id if turn.message else None}

    def _allows(self, ref, agents, now):
        """Apply current sender settings when scheduling and immediately before sending."""
        agent = agents.get(self.runtime.outbox.owners.get(ref))
        return agent is not None and agent.enabled and (agent.notify_window is None or agent.notify_window.allows(now))

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
            candidates = []
            for ref in self.pending.values():
                item = self.runtime.outbox.records[ref]
                if not self._allows(ref, agents, now):
                    continue
                if item.priority == "P0":
                    if (now - item.created_at).total_seconds() < self.config.p0_delay_seconds:
                        continue
                elif busy or idle < getattr(self.config, f"{item.priority.lower()}_idle_seconds"):
                    continue
                candidates.append(ref)
            if candidates:
                priority = min(self.runtime.outbox.records[ref].priority for ref in candidates)
                candidates.sort(key=lambda ref: (self.runtime.outbox.records[ref].created_at, ref.id))
                identifiers = [ref.id for ref in candidates if self.runtime.outbox.records[ref].priority == priority][
                    :20
                ]
                self.worker = asyncio.create_task(
                    self._wake(run_job, priority, identifiers),
                    name="inbox:chat",
                )

    async def _wake(self, run_job, priority, identifiers):
        permanent = False
        self.last_started_at = self.runtime.now()
        try:
            response = await run_job("inbox_chat", priority=priority, msg_ids=identifiers)
            if not response.success:
                failure = response.metadata.get("reme.job_error", {})
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
