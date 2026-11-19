"""Canonical Markdown reminders with delivery state in frontmatter."""

import asyncio
from datetime import date

import frontmatter
import yaml

from ....schema.agent_runtime import AgentReminder, ReminderRef
from ....schema.event import Event
from ....utils.async_utils import complete_in_thread, finish_cleanup


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
        self.issues: dict[str, str] = {}

    def _scan(self):
        records, owners, cache, issues = {}, {}, {}, {}
        for agent in self.runtime.agents.load(include_deleted=True).values():
            directory = self.runtime.agents.directory_for(agent) / "reminder"
            for path in sorted(directory.glob("*/*.md")):
                try:
                    if date.fromisoformat(path.parent.name).isoformat() != path.parent.name:
                        raise ValueError("reminder directories must use YYYY-MM-DD names")
                    self.files.checked(path)
                    stat = path.stat()
                    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    previous = self.cache.get(path)
                    item = previous[1] if previous and previous[0] == signature else self._read(path)
                except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
                    issues[path.relative_to(self.files.workspace).as_posix()] = str(exc)
                    continue
                cache[path] = (signature, item)
                if item is not None:
                    ref = ReminderRef(file=path.relative_to(self.files.workspace).as_posix(), id=item.id)
                    records[ref] = item
                    owners[ref] = agent.agent_name
        return records, owners, cache, issues

    def _read(self, path):
        document = frontmatter.loads(self.files.read(path))
        # Model drafts become canonical only after the runtime assigns an ID.
        if "id" not in document:
            return None
        metadata = dict(document.metadata)
        if metadata.pop("action", "write") != "write":
            raise ValueError("canonical reminders must have action write")
        return AgentReminder.model_validate({**metadata, "message": document.content})

    def _write(self, path, item, *, on_write=None):
        metadata = item.model_dump(mode="json", exclude={"message"})
        self.files.write(
            path,
            frontmatter.dumps(frontmatter.Post(item.message, action="write", **metadata)) + "\n",
            **({"on_write": lambda version: on_write(path, version)} if on_write is not None else {}),
        )

    @staticmethod
    def _committed_references(records, receipts, issues):
        committed = {
            ReminderRef.model_validate(ref)
            for group in receipts.values()
            for receipt in group
            for ref in receipt.metadata.get("reminders", [])
        }
        committed.update(ref for ref, item in records.items() if item.delivered_at is not None)
        for ref in committed - records.keys():
            issues.setdefault(ref.file, f"committed reminder is missing or invalid: {ref.id}")
        return committed & records.keys()

    async def sync(self):
        """Load complete records, repair reference views, and retry pending publication."""
        async with self.sync_lock:
            await self._sync()

    async def refresh(self):
        """Refresh canonical records without publishing or consuming messages."""
        receipts = await self.runtime.inbox.receipts()
        async with self.lock:
            records, owners, cache, issues = await asyncio.to_thread(self._scan)
            committed = self._committed_references(records, receipts, issues)
            for path, error in issues.items():
                if self.issues.get(path) != error:
                    self.runtime.logger.error(f"Reminder unavailable at {path}: {error}")
            self.records, self.owners, self.cache = records, owners, cache
            self.committed = committed
            self.issues = issues
        return receipts

    async def _sync(self):
        receipts = await self.refresh()
        await self.reconcile()
        try:
            if not self.runtime.closed:
                await self.publish()
        finally:
            for name, group in receipts.items():
                retained, released = [], []
                for receipt in group:
                    handed_off = all(
                        ReminderRef.model_validate(ref) in self.records
                        and self.records[ReminderRef.model_validate(ref)].delivered_at is not None
                        for ref in receipt.metadata.get("reminders", [])
                    )
                    (released if handed_off else retained).extend(receipt.item_ids)
                await self.runtime.inbox.retain(name, retained)
                await self.runtime.inbox.retain(name, released, release=True)

    async def reconcile(self):
        """Rebuild Chat reference views from canonical reminder acknowledgements."""
        pending, processed = [], []
        for ref in sorted(self.committed, key=lambda ref: ref.id):
            item = self.records[ref]
            event = self._event(ref)
            if item.read_at is not None:
                processed.append({"item": event, "consume_ts": item.read_at.isoformat()})
            elif item.delivered_at is not None:
                pending.append(event)
        await self.runtime.inbox.reconcile("Remy", pending, processed)

    def _event(self, ref):
        return Event(
            id=ref.id,
            event_ts=self.records[ref].created_at,
            source=self.owners[ref],
            description=f"Reminder from {self.owners[ref]}",
            paths=[ref.file],
        ).model_dump(mode="json")

    async def save(self, agent, run_id, result, now, *, on_write=None):
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
                await complete_in_thread(
                    self._write,
                    path,
                    item,
                    **({"on_write": on_write} if on_write is not None else {}),
                )
                refs.append(ReminderRef(file=path.relative_to(self.files.workspace).as_posix(), id=item.id))
                self.cache.pop(path, None)
        return refs

    async def update(self, ref, **changes):
        """Atomically update acknowledgement fields in the canonical Markdown."""
        if set(changes) - {"delivered_at", "read_at", "user_read_at", "chat_message_ids"}:
            raise ValueError("only reminder acknowledgements can be updated")
        async with self.lock:
            path = self.files.path(ref.file)
            item = await asyncio.to_thread(self._read, path)
            if item is None or item.id != ref.id:
                raise ValueError("reminder identity is missing or changed")
            message_ids = changes.pop("chat_message_ids", [])
            changes = {key: value for key, value in changes.items() if getattr(item, key) is None}
            if message_ids:
                combined = list(dict.fromkeys([*item.chat_message_ids, *message_ids]))
                if combined != item.chat_message_ids:
                    changes["chat_message_ids"] = combined
            if changes:
                item = AgentReminder.model_validate({**item.model_dump(), **changes})
                await complete_in_thread(self._write, path, item)
                self.cache.pop(path, None)
            if ref in self.records:
                self.records[ref] = item

    async def publish(self):
        """Stable references make receiver publication safe to repeat after interruption."""
        agents = await asyncio.to_thread(self.runtime.agents.load)
        async with self.lock:
            refs = sorted(self.committed, key=lambda ref: (self.records[ref].created_at, ref.id))
        for ref in refs:
            async with self.lock:
                item = self.records.get(ref)
                owner = self.owners.get(ref)
                agent = agents.get(owner)
                if ref not in self.committed or item is None or item.delivered_at is not None:
                    continue
                if agent is None or not agent.enabled:
                    continue
                event = self._event(ref)
            # Capacity waits must leave the outbox available to readers and consumers.
            await self.runtime.inbox.append("Remy", event)
            await self.update(ref, delivered_at=self.runtime.now())
            self.runtime.logger.info(
                f"Proactive agent={owner} reminder={ref.id} result=delivered path={ref.file}",
            )
            self.runtime.changed.set()

    def references(self, agent_name):
        """Find retained records owned by an identity without storing duplicate sender fields."""
        return {ref for ref, owner in self.owners.items() if owner == agent_name}

    async def list(self, agent_name, *, state="all", offset=0, limit=20):
        """Page one task's committed outgoing reminders."""
        if state not in {"all", "pending", "processed"}:
            raise ValueError("invalid mailbox state")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
        await self.runtime.definition(agent_name)
        await self.refresh()
        async with self.lock:
            refs = [ref for ref in self.committed if self.owners[ref] == agent_name]
            refs = [
                ref
                for ref in refs
                if state == "all" or (self.records[ref].read_at is not None) == (state == "processed")
            ]
            refs.sort(key=lambda ref: (self.records[ref].created_at, ref.id), reverse=True)
            return {
                "items": [self.message(ref) for ref in refs[offset : offset + limit]],
                "total": len(refs),
                "offset": offset,
                "limit": limit,
            }

    def message(self, ref):
        """Render canonical content for Inbox, Outbox and single-reminder views."""
        return {**self.records[ref].model_dump(mode="json"), "from_agent_name": self.owners[ref]}

    async def read(self, agent_name, message_id, *, mark_read=False):
        """Authorize the mailbox before updating only the user's read receipt."""
        if agent_name != "Remy":
            await self.runtime.definition(agent_name)
        await self.refresh()
        if agent_name == "Remy" and message_id.startswith("reme-proactive-"):
            async with self.lock:
                refs = [ref for ref in self.committed if message_id in self.records[ref].chat_message_ids]
            if not refs and not mark_read:
                raise ValueError("chat reminder has no retained source messages")

            async def acknowledge():
                for ref in refs:
                    await self.update(ref, user_read_at=self.runtime.now())

            if mark_read:
                await finish_cleanup(acknowledge())
            async with self.lock:
                return {"id": message_id, "items": [self.message(ref) for ref in refs]}
        async with self.lock:
            ref = next((ref for ref in self.committed if ref.id == message_id), None)
            if (
                ref is None
                or (agent_name == "Remy" and not (self.records[ref].delivered_at or self.records[ref].read_at))
                or (agent_name not in {"Remy", self.owners[ref]})
            ):
                raise ValueError("message not found in this mailbox")
        if mark_read:
            await self.update(ref, user_read_at=self.runtime.now())
        async with self.lock:
            return self.message(ref)
