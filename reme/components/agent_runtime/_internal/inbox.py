"""Chat Agent inbox views containing only canonical reminder references."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from hashlib import sha256
from uuid import uuid4

from ....schema.agent_runtime import InboxConfig, ReminderRef
from ....schema.agent_session import SessionMessage
from ....utils.async_utils import complete_in_thread, finish_cleanup


@dataclass
class InboxTurn:
    """One disposable fork's tools and explicit delivery decision."""

    session_id: str
    priority: str
    msg_ids: tuple[str, ...]
    chat_revision: int = 0
    session_revision: str | None = None
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


class AgentInbox:
    """Keep unread references bounded and all message state in the sender's reminder."""

    def __init__(self, runtime, config: InboxConfig, max_pending=1000):
        self.runtime = runtime
        self.config = config
        self.max_pending = max_pending
        self.lock = asyncio.Lock()
        self.pending: dict[str, ReminderRef] = {}
        self.read_refs: dict[str, ReminderRef] = {}
        self.last_activity_at = runtime.now()
        self.worker = None
        self.retry_at = None
        self.active_turn: InboxTurn | None = None
        self.chat_revision = 0
        self.error = None
        self.failures = 0
        self.blocked = False
        self.last_started_at = None
        self.last_completed_at = None

    def directory(self, agent_id="chat"):
        """Only chat receives notifications in the current messaging contract."""
        if agent_id != "chat":
            raise ValueError("only the Chat Agent has an inbox")
        return self.runtime.agents.path("chat")

    def path(self, read=False):
        """The files are rebuildable reference lists, not duplicate messages."""
        return self.directory() / ("inbox_read.jsonl" if read else "inbox_unread.jsonl")

    def activity(self):
        """Restart idle timing after every foreground or internally requested Chat turn."""
        self.last_activity_at = self.runtime.now()

    def foreground(self):
        """Invalidate an older reminder decision as soon as a user turn is queued."""
        self.chat_revision += 1

    def _refs(self, read):
        try:
            return {
                ReminderRef.model_validate(value)
                for value in self.runtime.files.iter_jsonl(self.path(read), incomplete_ok=True)
            }
        except ValueError as exc:
            self.runtime.logger.warning(f"Rebuilding invalid inbox view {self.path(read)}: {exc}")
            return set()

    async def _write_indexes(self):
        outbox = self.runtime.outbox
        unread, read = {}, {}
        for ref in outbox.committed:
            item = outbox.records[ref]
            target = read if item.read_at is not None else unread
            if item.delivered_at is not None or item.read_at is not None:
                if ref.id in target and target[ref.id] != ref:
                    raise ValueError(f"duplicate reminder ID: {ref.id}")
                target[ref.id] = ref
        if len(unread) > self.max_pending:
            raise ValueError("inbox exceeds its unread capacity")
        retained = sorted(read.values(), key=lambda ref: (outbox.records[ref].read_at, ref.id))[-1000:]
        await complete_in_thread(
            self.runtime.files.write_jsonl,
            self.path(True),
            [ref.model_dump() for ref in retained],
        )
        await complete_in_thread(
            self.runtime.files.write_jsonl,
            self.path(),
            [ref.model_dump() for ref in sorted(unread.values(), key=lambda ref: ref.id)],
        )
        self.pending, self.read_refs = unread, {ref.id: ref for ref in retained}

    async def reconcile(self):
        """Treat external read references as requests, then rebuild views from canonical receipts."""
        async with self.lock:
            read, unread = await asyncio.to_thread(lambda: (self._refs(True), self._refs(False)))
            outbox = self.runtime.outbox
            # Views are disposable. Missing/corrupt source records are reported by the
            # outbox and must not prevent unrelated messages from being processed.
            read &= outbox.committed
            unread &= outbox.committed
            for ref in unread:
                await outbox.update(ref, delivered_at=self.runtime.now())
            for ref in read:
                await outbox.update(ref, delivered_at=self.runtime.now(), read_at=self.runtime.now())
            await self._write_indexes()

    async def deliver(self, ref):
        """Persist the reference before acknowledging delivery to its sender."""
        async with self.lock:
            item = self.runtime.outbox.records[ref]
            if item.read_at is not None:
                return
            previous = self.pending.get(ref.id) or self.read_refs.get(ref.id)
            if previous is not None and previous != ref:
                raise ValueError(f"duplicate reminder ID: {ref.id}")
            if previous is None:
                if len(self.pending) >= self.max_pending:
                    raise TimeoutError("inbox is full; mark processed messages read first")
                await complete_in_thread(self.runtime.files.append_jsonl, self.path(), ref.model_dump())
            await self.runtime.outbox.update(ref, delivered_at=self.runtime.now())
            self.pending[ref.id] = ref
            self.runtime.changed.set()

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
                        "from_agent_id": outbox.owners[ref],
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
                await self._write_indexes()

            await finish_cleanup(save())

    @asynccontextmanager
    async def turn(self, session_id, priority, msg_ids):
        """Revoke the fork's tool scope on completion, failure and cancellation."""
        if self.active_turn is not None:
            raise RuntimeError("an inbox fork is already running")
        async with self.lock:
            selected = tuple(
                identifier
                for identifier in dict.fromkeys(msg_ids)
                if identifier in self.pending
                and self.runtime.outbox.records[self.pending[identifier]].priority == priority
            )
        if not selected:
            raise ValueError("no pending messages at the selected priority")
        revision = (await self.runtime.agent_wrapper.read_session(session_id)).revision
        turn = InboxTurn(session_id, priority, selected, chat_revision=self.chat_revision, session_revision=revision)
        self.active_turn = turn
        try:
            yield turn
        finally:
            self.active_turn = None

    def current_turn(self, identifier):
        """Reject foreground calls and capabilities left over from a finished fork."""
        if self.active_turn is None or identifier != self.active_turn.id:
            raise ValueError("inbox tools require an active reminder fork")
        return self.active_turn

    async def read_turn(self, turn, priority):
        """Expose only this wakeup's batch; later arrivals wait for another fork."""
        if priority != turn.priority:
            raise ValueError("priority does not match the active reminder fork")
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
                async with self.runtime.chat_lock:
                    current = await self.runtime.sessions.current("chat")
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
                        if content and turn.message is None:
                            turn.message = await self.runtime.agent_wrapper.append_session(
                                turn.session_id,
                                content,
                                message_id=turn.message_id,
                                role="assistant",
                                expected_revision=turn.session_revision,
                                delivery="immediate" if turn.priority == "P0" else "idle",
                            )
                        await self.consume([item["id"] for item in turn.items])
                        turn.done = True

                    await finish_cleanup(commit())
            return {"sent": bool(content), "message_id": turn.message.id if turn.message else None}

    def _allows(self, ref, agents, now):
        """Apply current sender settings when scheduling and immediately before sending."""
        agent = agents.get(self.runtime.outbox.owners.get(ref))
        return agent is not None and agent.enabled and (agent.notify_window is None or agent.notify_window.allows(now))

    async def tick(self, run_job, job_name):
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
                    self._wake(run_job, job_name, priority, identifiers),
                    name="inbox:chat",
                )

    async def _wake(self, run_job, job_name, priority, identifiers):
        permanent = False
        self.last_started_at = self.runtime.now()
        try:
            response = await run_job(job_name, priority=priority, msg_ids=identifiers)
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
