"""File-native sending intentions, owned by their Runtime."""

import asyncio
from abc import abstractmethod
import re

from ...base import BaseComponent
from ....enumeration import ComponentEnum
from ....utils.async_utils import complete_in_thread
from ....utils.files.path_lock import get_path_lock
from ....utils.files.workspace_files import WorkspaceFiles
from ..overview_events import notify_overview


class BaseOutbox(BaseComponent):
    """Persist immutable payloads before calling a receiver or native host."""

    component_type = ComponentEnum.AGENT_OUTBOX
    delivery_model = None
    terminal_states = frozenset()

    def __init__(self, directory, **kwargs):
        super().__init__(**kwargs)
        self.files = WorkspaceFiles(self.workspace_path)
        self.directory = self.files.path(directory)
        self.issues = {}
        self.agent_name = None
        self.inbox = None

    def path(self, identifier):
        """Delivery identities are filenames, never caller-selected paths."""
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", identifier):
            raise ValueError("Invalid delivery id")
        return self.files.checked(self.directory / f"{identifier}.json")

    async def _start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        await self.recover()

    async def _close(self):
        return None

    async def read(self, identifier):
        """Validate persisted state before using any delivery instruction."""
        path = self.path(identifier)
        value = await asyncio.to_thread(self.files.read_json, path, None)
        if value is None:
            return None
        delivery = self.delivery_model.model_validate(value)
        if delivery.id != identifier:
            raise ValueError("Delivery identity does not match its filename")
        return delivery

    async def save(self, delivery):
        """Callers hold the delivery lock across state transitions."""
        delivery = self.delivery_model.model_validate(delivery.model_dump())
        await complete_in_thread(self.files.write_json, self.path(delivery.id), delivery.model_dump(mode="json"))
        self.issues.pop(delivery.id, None)
        self._notify(delivery)
        return delivery

    def _notify(self, delivery):
        if self.agent_name is None:
            return
        notification = getattr(delivery, "notification", None)
        if notification is not None:
            if delivery.status == "delivered":
                notify_overview(self)
                return
            trace = notification.trace
            notify_overview(
                self,
                "notification_prepared",
                identity=delivery.id,
                agent_name=notification.source_agent,
                target_agent=delivery.target_agent,
                delivery_id=delivery.id,
                run_id=trace.run_id if trace else None,
                run_ids=[trace.run_id] if trace else [],
                source_id=trace.source_refs[0].source_id if trace and len(trace.source_refs) == 1 else None,
                event_ids=trace.event_ids if trace else [],
                paths=[notification.record_path],
                occurred_at=(delivery.delivered_at or delivery.created_at).isoformat(),
            )
        notify_overview(self)

    async def entries(self):
        """Isolate damaged records without hiding their health information."""
        values, issues = [], {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                value = await self.read(path.stem)
            except (OSError, ValueError) as exc:
                issues[path.stem] = str(exc)
            else:
                if value is not None:
                    values.append(value)
        self.issues = issues
        return values

    async def pending(self):
        """Return sending intentions whose delivery is not terminal."""
        return [value for value in await self.entries() if value.status not in self.terminal_states]

    async def prepare(self, delivery):
        """Return an existing identical intent, or publish a new one atomically."""
        delivery = self.delivery_model.model_validate(delivery.model_dump())
        async with await get_path_lock(self.path(delivery.id)):
            previous = await self.read(delivery.id)
            if previous is not None:
                mutable = {"status", "delivered_at", "acknowledged_at", "last_error"}
                if previous.model_dump(exclude=mutable) != delivery.model_dump(exclude=mutable):
                    raise ValueError("Conflicting delivery identity")
                return previous
            return await self.save(delivery)

    async def recover(self):
        """Load local intentions; workflows choose when to perform external I/O."""
        return await self.pending()

    @abstractmethod
    async def deliver(self, delivery_id):
        """Perform one recoverable sending operation."""
