"""Reminder references and their receiver-owned processing facts."""

from hashlib import sha256

from .base import BaseInbox
from ....core.registry import R
from ....schema.agent_inbox import NotificationRef
from ....schema.proactive_agent import ProactiveRecordDecision
from ....utils.proactive_record import split_record


@R.register("notification")
class NotificationInbox(BaseInbox):
    """Keep processing receipts even after they leave the displayed history window."""

    keep_processed = True
    overview_kind = "notification"

    def validate_item(self, value):
        """Validate and normalize one durable queue payload."""
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return NotificationRef.model_validate(value).model_dump(mode="json")

    def material(self, reference):
        """Read reminder content and its version from the same file snapshot."""
        reference = NotificationRef.model_validate(reference)
        text = self.files.read(self.files.path(reference.record_path))
        decision = ProactiveRecordDecision.parse(split_record(text)[0])
        return decision.reminder(reference.record_path), sha256(text.encode()).hexdigest()

    async def reference(self, identifier):
        """Find a reference without claiming or consuming it."""
        async with self.changed:
            self._check_open()
            if identifier in self._pending:
                return NotificationRef.model_validate(self._pending[identifier])
            for batch, _ in reversed(self._history):
                for item in batch:
                    if item["id"] == identifier:
                        return NotificationRef.model_validate(item)
        return None

    async def processed(self, identifier):
        """Return the durable receiver receipt for one notification."""
        for receipt in reversed(await self.receipts()):
            if identifier in receipt.item_ids:
                return receipt
        return None

    async def pending_references(self):
        """The scheduler chooses priority from source records, without claiming inputs."""
        async with self.changed:
            self._check_open()
            return [NotificationRef.model_validate(item) for item in self._pending.values()]
