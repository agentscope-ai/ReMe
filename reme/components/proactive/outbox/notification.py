"""Transfer committed record references to Remy's NotificationInbox."""

from hashlib import sha256

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

    async def prepare_reference(self, reference):
        """Freeze a stable record reference and its destination before handoff."""
        reference = NotificationRef.model_validate(reference)
        identifier = "delivery-" + sha256(f"{reference.id}\0{self.target.name}".encode()).hexdigest()
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
            await self.target.inbox.append(delivery.notification, wait=False)
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
