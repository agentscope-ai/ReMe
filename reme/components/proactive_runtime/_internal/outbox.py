"""Canonical Markdown reminders keyed by workspace-relative filename."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext

import yaml

from .reminder_files import ReminderFiles
from ....constants import CHAT_AGENT_NAME, PROACTIVE_MESSAGE_ID_PREFIX
from ....schema.proactive_runtime import AgentReminder
from ....utils.async_utils import complete_in_thread
from ....utils.files.path_lock import get_path_lock

_FILE_ERRORS = (OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError)


class ReminderOutbox:
    """Deliver committed reminders and rebuild the inbox from their files."""

    def __init__(self, runtime, daily_dir):
        self.runtime = runtime
        self.files = runtime.files
        self.storage = ReminderFiles(self.files, daily_dir)
        self.lock = asyncio.Lock()
        self.sync_lock = asyncio.Lock()
        self.records: dict[str, AgentReminder] = {}
        self.owners: dict[str, str] = {}
        self.cache = {}
        self.issues: dict[str, str] = {}

    def _scan(self):
        records, owners, cache, issues = {}, {}, {}, {}
        for agent in self.runtime.agents.load(include_deleted=True).values():
            for path in self.storage.paths(agent.agent_name):
                ref = path.relative_to(self.files.workspace).as_posix()
                try:
                    self.files.checked(path)
                    stat = path.stat()
                    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                    previous = self.cache.get(path)
                    item = previous[1] if previous and previous[0] == signature else self.storage.read(path)
                except _FILE_ERRORS as exc:
                    issues[ref] = str(exc)
                    continue
                cache[path] = (signature, item)
                if item is not None:
                    records[ref] = item
                    owners[ref] = agent.agent_name
        return records, owners, cache, issues

    def _write(self, path, item):
        self.files.write(path, self.storage.serialize(path, item))

    def _report(self, ref, error):
        if self.issues.get(ref) != error:
            self.runtime.logger.error(f"Reminder unavailable at {ref}: {error}")
            self.issues[ref] = error

    async def refresh(self):
        """Reload delivered records, leaving uncommitted files out of mailbox views."""
        receipts = await self.runtime.inbox.receipts()
        async with self.lock:
            records, owners, cache, issues = await asyncio.to_thread(self._scan)
            for group in receipts.values():
                for receipt in group:
                    ref = receipt.metadata.get("reminder_path")
                    if isinstance(ref, str) and ref not in records and ref in self.issues:
                        issues.setdefault(ref, self.issues[ref])
            for ref, error in issues.items():
                self._report(ref, error)
            self.records, self.owners, self.cache, self.issues = records, owners, cache, issues
        return receipts

    async def sync(self):
        """Recover delivery from durable input receipts before releasing their retention."""
        async with self.sync_lock:
            receipts = await self.refresh()
            for agent, group in receipts.items():
                retained, released = [], []
                for receipt in group:
                    ref = receipt.metadata.get("reminder_path")
                    if self.storage.owns(ref, agent) and ref not in self.records:
                        path = self.files.workspace / ref
                        async with await get_path_lock(path):
                            await self._deliver_locked(agent, ref, path)
                        if ref not in self.records:
                            retained.extend(receipt.item_ids)
                            continue
                    released.extend(receipt.item_ids)
                await self.runtime.inbox.retain(agent, retained)
                await self.runtime.inbox.retain(agent, released, release=True)

    def order(self, ref):
        """Order reminders by priority, delivery time and stable path."""
        item = self.records[ref]
        return item.priority, item.delivered_at, ref

    def allows(self, ref, agents, now):
        """Apply the sender's settings when Chat is about to handle a reminder."""
        agent = agents.get(self.owners.get(ref))
        return agent is not None and agent.enabled and (agent.notify_window is None or agent.notify_window.allows(now))

    def event(self, ref):
        """Describe a reminder without limiting its path to a generic event ID's length."""
        owner = self.owners[ref]
        return {
            "id": ref,
            "event_ts": self.records[ref].delivered_at.isoformat(),
            "source": owner,
            "description": f"Reminder from {owner}",
            "paths": [ref],
            "metadata": {},
        }

    @asynccontextmanager
    async def handoff(self, agent_name, reminder, record):
        """Hold the record's write lock from validation through the receipt and initial delivery."""
        ref = None
        path = record.path if record is not None else None
        if reminder is not None:
            ref = reminder.id
            if not self.storage.owns(ref, agent_name):
                raise ValueError("a reminder must reference the task's dated daily record")
            path = self.files.path(ref)
        async with await get_path_lock(path) if path is not None else nullcontext():
            if record is not None:
                await asyncio.to_thread(record.check_version, path)
            if ref is not None:
                draft = await asyncio.to_thread(self.storage.read, path, draft=True)
                if draft.delivered_at is not None:
                    raise ValueError(f"cannot overwrite a delivered reminder: {path}")
                if (draft.priority, draft.message) != (reminder.priority, reminder.message):
                    raise ValueError(f"reminder decision changed: {path}")
            yield ref
            if ref is not None:
                await self._deliver_locked(agent_name, ref, path)

    async def _deliver_locked(self, agent, ref, path):
        try:
            item = await asyncio.to_thread(self.storage.read, path, draft=True)
            if item.delivered_at is None:
                item = item.model_copy(update={"delivered_at": self.runtime.now()})
                await complete_in_thread(self._write, path, item)
            async with self.lock:
                self.records[ref] = item
                self.owners[ref] = agent
                self.cache.pop(path, None)
                self.issues.pop(ref, None)
        except _FILE_ERRORS as exc:
            async with self.lock:
                self._report(ref, str(exc))

    async def update(self, ref, **changes):
        """Update Chat receipts under the workspace's file lock."""
        async with await get_path_lock(self.files.path(ref)):
            await self._update_locked(ref, **changes)

    @asynccontextmanager
    async def decision(self, items):
        """Keep the reminder content stable through native delivery and acknowledgement."""
        paths = {item["id"]: self.files.path(item["id"]) for item in items}
        async with AsyncExitStack() as stack:
            for path in sorted(paths.values()):
                await stack.enter_async_context(await get_path_lock(path))
            for expected in items:
                latest = await asyncio.to_thread(self.storage.read, paths[expected["id"]])
                if (
                    latest is None
                    or latest.read_at is not None
                    or (latest.priority, latest.message) != (expected["priority"], expected["message"])
                ):
                    raise RuntimeError("reminders changed; reassess the current inbox before submitting")
            yield self._update_locked

    async def _update_locked(self, ref, *, read_at=None, chat_message_ids=()):
        async with self.lock:
            path = self.files.path(ref)
            item = await asyncio.to_thread(self.storage.read, path)
            if item is None:
                raise ValueError(f"reminder is missing or undelivered: {ref}")
            updated = AgentReminder.model_validate(
                {
                    **item.model_dump(),
                    "read_at": item.read_at or read_at,
                    "chat_message_ids": list(dict.fromkeys([*item.chat_message_ids, *chat_message_ids])),
                },
            )
            if updated != item:
                await complete_in_thread(self._write, path, updated)
                self.cache.pop(path, None)
            if ref in self.records:
                self.records[ref] = updated

    def processed_refs(self):
        """Keep the same bounded read history for Chat acknowledgements and the inbox view."""
        return sorted(
            (ref for ref, item in self.records.items() if item.read_at is not None),
            key=lambda ref: (self.records[ref].read_at, ref),
        )[-self.runtime.inbox.max_read_items :]

    async def inbox(self, *, state="all", offset=0, limit=20):
        """Project the receiver view directly from daily records."""
        if state not in {"all", "pending", "processing", "processed"}:
            raise ValueError("invalid item state")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
        await self.refresh()
        turn = self.runtime.notifications.active_turn
        claimed = set(turn.msg_ids) if turn else set()
        async with self.lock:
            pending = [ref for ref, item in self.records.items() if item.read_at is None]
            items = []
            for ref in [*self.processed_refs(), *pending]:
                item = self.records[ref]
                status = "processed" if item.read_at is not None else "processing" if ref in claimed else "pending"
                if state in {"all", status}:
                    items.append(
                        {
                            "event": self.event(ref),
                            "state": status,
                            "consume_ts": item.read_at.isoformat() if item.read_at else None,
                            "reminder": self.message(ref),
                        },
                    )
            items.sort(key=lambda value: (value["event"]["event_ts"], value["event"]["id"]), reverse=True)
            return {"items": items[offset : offset + limit], "total": len(items), "offset": offset, "limit": limit}

    async def list(self, agent_name, *, state="all", offset=0, limit=20):
        """Page one task's delivered outgoing reminders."""
        if state not in {"all", "pending", "processed"}:
            raise ValueError("invalid mailbox state")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
        await self.runtime.definition(agent_name)
        await self.refresh()
        async with self.lock:
            refs = [
                ref
                for ref, item in self.records.items()
                if self.owners[ref] == agent_name
                and (state == "all" or (item.read_at is not None) == (state == "processed"))
            ]
            refs.sort(key=lambda ref: (self.records[ref].delivered_at, ref), reverse=True)
            return {
                "items": [self.message(ref) for ref in refs[offset : offset + limit]],
                "total": len(refs),
                "offset": offset,
                "limit": limit,
            }

    def message(self, ref):
        """Render the source path as the reminder ID."""
        return {**self.records[ref].model_dump(mode="json"), "from_agent_name": self.owners[ref]}

    async def read(self, agent_name, message_id):
        """Read a task reminder, or find the sources of a combined Chat message."""
        if agent_name != CHAT_AGENT_NAME:
            await self.runtime.definition(agent_name)
        await self.refresh()
        async with self.lock:
            if message_id in self.records:
                if agent_name in {CHAT_AGENT_NAME, self.owners[message_id]}:
                    return self.message(message_id)
            elif agent_name == CHAT_AGENT_NAME and message_id.startswith(PROACTIVE_MESSAGE_ID_PREFIX):
                refs = sorted(ref for ref, item in self.records.items() if message_id in item.chat_message_ids)
                if not refs:
                    raise ValueError("chat reminder has no retained source messages")
                return {"id": message_id, "items": [self.message(ref) for ref in refs]}
            raise ValueError("message not found in this mailbox")
