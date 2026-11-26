"""Recoverable assistant-message delivery into an existing native session."""

from datetime import datetime, timezone

from .base import BaseOutbox
from ....constants import PROACTIVE_MESSAGE_ID_PREFIX
from ....core.registry import R
from ....schema.agent_outbox import ChatDelivery
from ....utils.files.path_lock import get_path_lock


@R.register("chat")
class ChatOutbox(BaseOutbox):
    """Keep the reply frozen through uncertain native writes and receiver ack."""

    delivery_model = ChatDelivery
    terminal_states = frozenset({"acknowledged", "aborted"})

    def __init__(self, wrapper, **kwargs):
        super().__init__(**kwargs)
        self.wrapper = wrapper

    async def find_message(self, delivery):
        """A lost response must be resolved before attempting another native write."""
        snapshot = await self.wrapper.read_session(delivery.session_id)
        identities = {delivery.message_id, PROACTIVE_MESSAGE_ID_PREFIX + delivery.message_id}
        return next((message for message in snapshot.messages if message.id in identities), None)

    async def deliver(self, delivery_id, *, validate=None):
        async with await get_path_lock(self.path(delivery_id)):
            delivery = await self.read(delivery_id)
            if delivery is None or delivery.status == "aborted":
                raise ValueError("Chat delivery is unavailable")
            if delivery.status in {"sent", "acknowledged"}:
                return delivery
            message = await self.find_message(delivery)
            if message is None:
                if validate is not None:
                    await validate()
                message = await self.wrapper.append_session(
                    delivery.session_id,
                    delivery.content,
                    message_id=delivery.message_id,
                    role="assistant",
                    expected_turn_id=delivery.session_turn_id,
                    expected_revision=delivery.session_revision if delivery.session_turn_id is None else None,
                    delivery="immediate" if delivery.session_turn_id is not None else "idle",
                )
            if message.content != delivery.content:
                raise ValueError("Native message conflicts with the prepared Chat reply")
            return await self.save(
                ChatDelivery.model_validate(
                    {
                        **delivery.model_dump(),
                        "status": "sent",
                        "delivered_at": datetime.now(timezone.utc),
                        "last_error": None,
                    },
                ),
            )

    async def acknowledge(self, identifier, now):
        """Finalize sender state after the receiver receipt is durable."""
        async with await get_path_lock(self.path(identifier)):
            delivery = await self.read(identifier)
            if delivery is None or delivery.status not in {"sent", "acknowledged"}:
                raise ValueError("Chat reply has not been sent")
            return await self.save(
                ChatDelivery.model_validate(
                    {
                        **delivery.model_dump(),
                        "status": "acknowledged",
                        "acknowledged_at": now,
                    },
                ),
            )

    async def abort(self, identifier, error):
        """Retire an unsent intention so a new decision can be made."""
        async with await get_path_lock(self.path(identifier)):
            delivery = await self.read(identifier)
            if delivery is not None and delivery.status == "prepared":
                return await self.save(
                    ChatDelivery.model_validate(
                        {
                            **delivery.model_dump(),
                            "status": "aborted",
                            "last_error": str(error),
                        },
                    ),
                )
        return delivery
