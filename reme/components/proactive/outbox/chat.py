"""Recoverable assistant-message delivery into an existing native session."""

from datetime import datetime, timezone

from .base import BaseOutbox
from ..overview_events import notify_overview
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
            fresh = message is None
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
            saved = await self.save(
                ChatDelivery.model_validate(
                    {
                        **delivery.model_dump(),
                        "status": "sent",
                        "delivered_at": datetime.now(timezone.utc),
                        "last_error": None,
                    },
                ),
            )
            if fresh:
                await self._notify_sent(saved)
            return saved

    async def _notify_sent(self, delivery):
        """A recovered native message updates state without replaying its old action."""
        if self.agent_name is None:
            return
        traces = []
        if self.inbox is not None:
            for identifier in delivery.notification_ids:
                reference = await self.inbox.reference(identifier)
                if reference is not None and reference.trace is not None:
                    traces.append(reference.trace)
        run_ids = list(dict.fromkeys(trace.run_id for trace in traces))
        source_ids = {ref.source_id for trace in traces for ref in trace.source_refs}
        notify_overview(
            self,
            "reminder_sent",
            identity=delivery.message_id,
            agent_name=self.agent_name,
            delivery_id=delivery.id,
            message_id=PROACTIVE_MESSAGE_ID_PREFIX + delivery.message_id,
            run_id=run_ids[0] if len(run_ids) == 1 else None,
            run_ids=run_ids,
            source_id=next(iter(source_ids)) if len(source_ids) == 1 else None,
            event_ids=list(dict.fromkeys(identifier for trace in traces for identifier in trace.event_ids)),
            paths=list(delivery.record_versions),
            occurred_at=delivery.delivered_at.isoformat(),
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
