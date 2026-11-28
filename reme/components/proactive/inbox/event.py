"""Observed events for one background Runtime."""

from datetime import datetime

from .base import BaseInbox
from ....core.registry import R
from ....schema.event import Event


@R.register("event")
class EventInbox(BaseInbox):
    """Validate the observation contract and order inputs by observation time."""

    overview_kind = "event"

    def validate_item(self, value):
        """Validate and normalize one durable queue payload."""
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return Event.model_validate(value).model_dump(mode="json")

    @staticmethod
    def sort_key(item):
        """Keep producer order unless a typed Inbox supplies an ordering."""
        return datetime.fromisoformat(item["event_ts"]), item["id"]
