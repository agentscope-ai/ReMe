"""Registered reminders keyed by their workspace-relative source path."""

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
    """Register committed reminders and expose their sender and receiver views."""

    def __init__(self, runtime, daily_dir):
        self.runtime = runtime
        self.files = runtime.files
        self.storage = ReminderFiles(self.files, daily_dir, runtime.agents.directory)
        self.lock = asyncio.Lock()
        self.sync_lock = asyncio.Lock()
        self.records: dict[str, AgentReminder] = {}
        self.owners: dict[str, str] = {}
        self.cache = {}
        self.issues: dict[str, str] = {}

    def _scan(self):
        records, owners, cache, issues = {}, {}, {}, {}
        for agent in self.runtime.agents.load(include_deleted=True).values():
            for path, state in self.storage.paths(agent.agent_name):
                ref = path.relative_to(self.files.workspace).as_posix()
                try:
                    stats = (self.files.checked(candidate).stat() for candidate in (path, state))
                    signature = tuple((stat.st_ino, stat.st_size, stat.st_mtime_ns) for stat in stats)
                    previous = self.cache.get(path)
                    item = (
                        previous[1]
                        if previous and previous[0] == signature
                        else self.storage.read(
                            agent.agent_name,
                            path,
                        )
                    )
                except _FILE_ERRORS as exc:
                    issues[ref] = str(exc)
                    continue
                cache[path] = (signature, item)
                if item is not None:
                    records[ref] = item
                    owners[ref] = agent.agent_name
        return records, owners, cache, issues

    def _write(self, path, item):
        self.storage.write(self.owners[item.id], path, item)

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
                            registered = await self._deliver_locked(agent, ref, path)
                        if not registered:
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

    async def receipt_record_paths(self, agent_name) -> frozenset[str]:
        """Get the paths that durable input receipts can still use for recovery."""
        receipts = await self.runtime.inbox.receipts()
        return frozenset(
            ref
            for receipt in receipts.get(agent_name, [])
            if isinstance(ref := receipt.metadata.get("reminder_path"), str)
        )

    def has_registration(self, agent_name, path):
        """Check whether the outbox reserves this path even if its source is missing."""
        return self.storage.state_path(agent_name, path).exists()

    @asynccontextmanager
    async def handoff(self, agent_name, reminder, record):
        """Hold the file lock from validation through input acknowledgement and registration."""
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
                if await asyncio.to_thread(self.has_registration, agent_name, path):
                    raise ValueError(f"cannot overwrite a delivered reminder: {path}")
                decision = await asyncio.to_thread(self.storage.decision, path)
                if decision.reminder(ref) != reminder:
                    raise ValueError(f"reminder decision changed: {path}")
            yield ref
            if ref is not None:
                await self._deliver_locked(agent_name, ref, path)

    async def _deliver_locked(self, agent, ref, path):
        try:
            state = self.storage.state_path(agent, path)
            if await asyncio.to_thread(state.is_file):
                return True
            decision = await asyncio.to_thread(self.storage.decision, path)
            item = decision.reminder(ref)
            if item is None:
                raise ValueError(f"Committed record no longer requests a reminder: {path}")
            item = item.model_copy(update={"delivered_at": self.runtime.now()})
            await complete_in_thread(self.storage.write, agent, path, item)
            async with self.lock:
                self.records[ref] = item
                self.owners[ref] = agent
                self.cache.pop(path, None)
                self.issues.pop(ref, None)
            return True
        except _FILE_ERRORS as exc:
            async with self.lock:
                self._report(ref, str(exc))
            return False

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
                ref = expected["id"]
                owner = self.owners.get(ref)
                latest = await asyncio.to_thread(self.storage.read, owner, paths[ref]) if owner is not None else None
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
            owner = self.owners.get(ref)
            item = await asyncio.to_thread(self.storage.read, owner, path) if owner is not None else None
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

    async def list(self, agent_name=None, *, state="all", offset=0, limit=20):
        """Page one task's outgoing reminders, or Chat's inbox when no sender is specified."""
        states = {"all", "pending", "processed"}
        if agent_name is None:
            states.add("processing")
        if state not in states:
            raise ValueError("invalid mailbox state")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
        if agent_name is not None:
            await self.runtime.definition(agent_name)
        await self.refresh()
        async with self.lock:
            turn = self.runtime.notifications.active_turn
            claimed = set(turn.msg_ids) if agent_name is None and turn else set()
            if agent_name is None:
                refs = [
                    *self.processed_refs(),
                    *(ref for ref, item in self.records.items() if item.read_at is None),
                ]
            else:
                refs = [ref for ref, owner in self.owners.items() if owner == agent_name]
            rows = []
            for ref in refs:
                item = self.records[ref]
                status = "processed" if item.read_at is not None else "processing" if ref in claimed else "pending"
                if state in {"all", status}:
                    rows.append((item.delivered_at, ref, status))
            rows.sort(reverse=True)
            return {
                "items": [{**self.message(ref), "state": status} for _, ref, status in rows[offset : offset + limit]],
                "total": len(rows),
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
