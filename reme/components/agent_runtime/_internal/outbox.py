"""Canonical Markdown reminders with delivery state in frontmatter."""

import asyncio
from datetime import date

import frontmatter

from ....schema.agent_runtime import AgentReminder, ReminderRef
from ....utils.async_utils import complete_in_thread


class AgentOutbox:
    """Rebuild messages from reminder files and publication eligibility from input receipts."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.files = runtime.files
        self.lock = asyncio.Lock()
        self.sync_lock = asyncio.Lock()
        self.records: dict[ReminderRef, AgentReminder] = {}
        self.owners: dict[ReminderRef, str] = {}
        self.committed: set[ReminderRef] = set()
        self.cache = {}

    def _scan(self):
        records, owners, cache = {}, {}, {}
        for agent in self.runtime.agents.load(include_deleted=True).values():
            directory = self.runtime.agents.directory_for(agent) / "reminder"
            if any(directory.glob("*.jsonl")):
                raise ValueError("daily reminder JSONL logs require migration to Markdown frontmatter")
            for path in sorted(directory.glob("*/*.md")):
                if date.fromisoformat(path.parent.name).isoformat() != path.parent.name:
                    raise ValueError("reminder directories must use YYYY-MM-DD names")
                self.files.checked(path)
                stat = path.stat()
                signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                previous = self.cache.get(path)
                item = previous[1] if previous and previous[0] == signature else self._read(path)
                cache[path] = (signature, item)
                if item is not None:
                    ref = ReminderRef(file=path.relative_to(self.files.workspace).as_posix(), id=item.id)
                    records[ref] = item
                    owners[ref] = agent.agent_id
        return records, owners, cache

    def _read(self, path):
        document = frontmatter.loads(self.files.read(path))
        # Model drafts become canonical only after the runtime assigns an ID.
        if "id" not in document:
            return None
        return AgentReminder.model_validate({**document.metadata, "message": document.content})

    def _write(self, path, item):
        metadata = item.model_dump(mode="json", exclude={"message"})
        self.files.write(path, frontmatter.dumps(frontmatter.Post(item.message, **metadata)) + "\n")

    @staticmethod
    def _committed_references(records, receipts):
        committed = {
            ReminderRef.model_validate(ref)
            for group in receipts.values()
            for receipt in group
            for ref in receipt.metadata.get("reminders", [])
        }
        committed.update(ref for ref, item in records.items() if item.delivered_at is not None)
        if committed - records.keys():
            raise ValueError("committed reminder is missing from its canonical file")
        return committed

    async def sync(self):
        """Load complete records, repair reference views, and retry pending publication."""
        async with self.sync_lock:
            await self._sync()

    async def _sync(self):
        receipts = await self.runtime.event_stream.receipts()
        async with self.lock:
            records, owners, cache = await asyncio.to_thread(self._scan)
            committed = self._committed_references(records, receipts)
            self.records, self.owners, self.cache = records, owners, cache
            self.committed = committed
        await self.runtime.inbox.reconcile()
        try:
            if not self.runtime.closed:
                await self.publish()
        finally:
            for name, group in receipts.items():
                retained, released = [], []
                for receipt in group:
                    handed_off = all(
                        self.records[ReminderRef.model_validate(ref)].delivered_at is not None
                        for ref in receipt.metadata.get("reminders", [])
                    )
                    (released if handed_off else retained).extend(receipt.event_ids)
                await self.runtime.event_stream.retain(name, retained)
                await self.runtime.event_stream.retain(name, released, release=True)

    async def save(self, agent, run_id, result, now):
        """Save validated model output before its event consumption receipt is committed."""
        directory = self.runtime.agents.directory_for(agent) / "reminder"
        refs = []
        async with self.lock:
            for index, notification in enumerate(result.notifications):
                path = self.files.path(notification.paths[0])
                if path.parent.parent != directory or path.suffix != ".md":
                    raise ValueError("the first notification path must be the Agent's dated reminder Markdown")
                if date.fromisoformat(path.parent.name).isoformat() != path.parent.name:
                    raise ValueError("reminder directories must use YYYY-MM-DD names")
                item = AgentReminder(**notification.model_dump(), id=f"{run_id}-{index}", created_at=now)
                previous = await asyncio.to_thread(self._read, path)
                if previous is not None:
                    if previous.id != item.id:
                        raise ValueError("cannot overwrite another reminder")
                    item = previous
                await complete_in_thread(self._write, path, item)
                refs.append(ReminderRef(file=path.relative_to(self.files.workspace).as_posix(), id=item.id))
                self.cache.pop(path, None)
        return refs

    async def update(self, ref, **changes):
        """Atomically update acknowledgement fields in the canonical Markdown."""
        if set(changes) - {"delivered_at", "woken_at", "read_at"}:
            raise ValueError("only reminder acknowledgements can be updated")
        async with self.lock:
            path = self.files.path(ref.file)
            item = await asyncio.to_thread(self._read, path)
            if item is None or item.id != ref.id:
                raise ValueError("reminder identity is missing or changed")
            changes = {key: value for key, value in changes.items() if getattr(item, key) is None}
            if changes:
                item = AgentReminder.model_validate({**item.model_dump(), **changes})
                await complete_in_thread(self._write, path, item)
                self.cache.pop(path, None)
            self.records[ref] = item

    async def publish(self):
        """Stable references make receiver publication safe to repeat after interruption."""
        agents = await asyncio.to_thread(self.runtime.agents.load)
        for ref in sorted(self.committed, key=lambda item: (self.records[item].created_at, item.id)):
            item = self.records[ref]
            agent = agents.get(self.owners[ref])
            if item.delivered_at is None and agent is not None and agent.enabled:
                await self.runtime.inbox.deliver(ref)

    def references(self, agent_id):
        """Find retained records owned by an identity without storing duplicate sender fields."""
        return {ref for ref, owner in self.owners.items() if owner == agent_id}
