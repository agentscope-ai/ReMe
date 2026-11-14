"""Main Agent inbox views containing only canonical reminder references."""

import asyncio
from datetime import timedelta

from ....schema.agent_runtime import InboxConfig, ReminderRef
from ....utils.async_utils import complete_in_thread, finish_cleanup


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

    def directory(self, agent_id="main"):
        """Only main receives notifications in the current messaging contract."""
        if agent_id != "main":
            raise ValueError("only the main Agent has an inbox")
        return self.runtime.agents.path("main")

    def path(self, read=False):
        """The files are rebuildable reference lists, not duplicate messages."""
        return self.directory() / ("inbox_read.jsonl" if read else "inbox_unread.jsonl")

    def activity(self):
        """Restart idle timing after every foreground or internally requested Chat turn."""
        self.last_activity_at = self.runtime.now()

    def _refs(self, read):
        return {
            ReminderRef.model_validate(value)
            for value in self.runtime.files.iter_jsonl(self.path(read), incomplete_ok=True)
        }

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
            if (read | unread) - outbox.committed:
                raise ValueError("inbox references an unknown or uncommitted reminder")
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

    async def read(self, agent_id, priority=None, unread_only=True, limit=20):
        """Expand references for callers without changing the read state."""
        self.directory(agent_id)
        async with self.lock:
            refs = list(self.pending.values())
            if not unread_only:
                refs.extend(self.read_refs.values())
            outbox = self.runtime.outbox
            refs = [ref for ref in refs if priority is None or outbox.records[ref].priority == priority]
            refs.sort(key=lambda ref: (outbox.records[ref].priority, outbox.records[ref].created_at, ref.id))
            return {
                "items": [
                    {
                        **outbox.records[ref].model_dump(mode="json", exclude={"woken_at"}),
                        "from_agent_id": outbox.owners[ref],
                        "to_agent_id": "main",
                        "reminder_path": ref.file,
                    }
                    for ref in refs[:limit]
                ],
            }

    async def mark_read(self, agent_id, msg_ids):
        """Update the one canonical read timestamp before moving its reference."""
        self.directory(agent_id)
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
            return {"marked_read": identifiers}

    async def tick(self, run_job, job_name):
        """Coalesce eligible messages into one lifecycle-owned Chat wakeup."""
        if self.worker is not None or self.runtime.closed:
            return
        await self.runtime.outbox.sync()
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
                agent = agents.get(self.runtime.outbox.owners[ref])
                if agent is None or not agent.enabled:
                    continue
                if agent.notify_window is not None and not agent.notify_window.allows(now):
                    continue
                if item.priority == "P0":
                    if item.woken_at is not None:
                        continue
                    if (now - item.created_at).total_seconds() < self.config.p0_delay_seconds:
                        continue
                elif busy or idle < getattr(self.config, f"{item.priority.lower()}_idle_seconds"):
                    continue
                candidates.append(ref)
            if candidates:
                priority = min(self.runtime.outbox.records[ref].priority for ref in candidates)
                identifiers = [ref.id for ref in candidates if self.runtime.outbox.records[ref].priority == priority]
                self.worker = asyncio.create_task(
                    self._wake(run_job, job_name, priority, identifiers),
                    name="inbox:main",
                )

    async def _wake(self, run_job, job_name, priority, identifiers):
        try:
            response = await run_job(job_name, priority=priority, msg_ids=identifiers)
            if not response.success:
                raise RuntimeError(str(response.answer))
            if isinstance(response.answer, dict) and response.answer.get("skipped"):
                if response.answer["skipped"] == "busy":
                    self.retry_at = self.runtime.now() + timedelta(seconds=self.runtime.retry_seconds)
                return
            async with self.lock:
                for identifier in identifiers:
                    ref = self.pending.get(identifier)
                    if ref is not None:
                        await self.runtime.outbox.update(ref, woken_at=self.runtime.now())
            self.retry_at = None
        except Exception as exc:
            self.retry_at = self.runtime.now() + timedelta(seconds=self.runtime.retry_seconds)
            self.runtime.logger.exception(f"Inbox wakeup failed: {exc}")
        finally:
            self.worker = None

    async def close(self):
        """Cancel the owned wakeup before closing shared agent infrastructure."""
        if self.worker is not None:
            task = self.worker
            task.cancel()
            await finish_cleanup(asyncio.gather(task, return_exceptions=True))
            self.worker = None
