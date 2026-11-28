"""Transfer committed record references to Remy's NotificationInbox."""

import asyncio

from .base import BaseOutbox
from ....core.registry import R
from ....schema.agent_inbox import NotificationRef
from ....schema.agent_outbox import NotificationDelivery
from ....utils.files.path_lock import get_path_lock


@R.register("notification")
class NotificationOutbox(BaseOutbox):
    """Sender delivery state never contains receiver read or processing state."""

    delivery_model = NotificationDelivery
    terminal_states = frozenset({"delivered"})

    def __init__(self, target, **kwargs):
        super().__init__(**kwargs)
        self.target = target

    @staticmethod
    def directory_for(agent_directory):
        """Locate retained notification deliveries beneath their source Agent."""
        return agent_directory / "outbox/notifications"

    def reference_path(self, notification_id):
        """Locate a notification's delivery even when its contents are damaged."""
        return self.path(NotificationDelivery.identity(notification_id, self.target.name))

    @classmethod
    async def read_reference(cls, files, agent_directory, notification_id, target_agent):
        """Read retained sender state without creating or starting its Runtime."""
        identifier = NotificationDelivery.identity(notification_id, target_agent)
        path = files.checked(cls.directory_for(agent_directory) / f"{identifier}.json")
        value = await asyncio.to_thread(files.read_json, path, None)
        if value is None:
            return None
        delivery = NotificationDelivery.model_validate(value)
        if delivery.id != identifier:
            raise ValueError("Delivery identity does not match its filename")
        return delivery

    async def prepare_reference(self, reference):
        """Freeze a stable record reference and its destination before handoff."""
        reference = NotificationRef.model_validate(reference)
        identifier = NotificationDelivery.identity(reference.id, self.target.name)
        return await self.prepare(
            NotificationDelivery(
                id=identifier,
                notification=reference,
                target_agent=self.target.name,
                created_at=reference.created_at,
            ),
        )

    async def deliver(self, delivery_id):
        async with await get_path_lock(self.path(delivery_id)):
            delivery = await self.read(delivery_id)
            if delivery is None:
                raise ValueError("Unknown notification delivery")
            if delivery.target_agent != self.target.name:
                raise ValueError("Notification target changed")
            if delivery.status == "delivered":
                return delivery
            try:
                await self.target.inbox.append(delivery.notification, wait=False)
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                await self.save(delivery.model_copy(update={"last_error": str(exc)[:2000]}))
                raise
            return await self.save(
                NotificationDelivery.model_validate(
                    {
                        **delivery.model_dump(),
                        "status": "delivered",
                        "delivered_at": self.target.now(),
                        "last_error": None,
                    },
                ),
            )
