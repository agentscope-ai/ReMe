"""Shared append validation and application-owned pending receipts.

Submission is distinct from persistence. A caller timing out never cancels a
queued append; the session owner settles it or fails it during its own cleanup.
Backend queues are consumed by their existing execution lifecycle, not by an
untracked background worker.
"""

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any

from ...schema import SessionMessage, SessionSnapshot


def validate_append(content, message_id, role, timeout, delivery):
    """Keep the public append contract identical across all three backends."""
    if role not in ("user", "assistant"):
        raise ValueError("role must be 'user' or 'assistant'")
    if not isinstance(message_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", message_id):
        raise ValueError("message_id must contain 1-128 letters, digits, underscores or hyphens")
    if not isinstance(content, str) or not content.strip() or len(content) > 100_000:
        raise ValueError("content must contain 1-100000 characters")
    if delivery not in ("idle", "immediate"):
        raise ValueError("Invalid chat delivery policy")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")


def check_identity(message, content, role):
    """A retry must refer to exactly the same observation."""
    if (message.content, message.role) != (content, role):
        raise ValueError("message_id already belongs to different content or role")


def check_guard(snapshot: SessionSnapshot, expected_turn_id, expected_revision):
    """Apply guards only to new appends, after recovering accepted retry IDs."""
    if expected_turn_id is not None and (snapshot.last_turn_id or "") != expected_turn_id:
        raise RuntimeError("Session changed before appending")
    if expected_revision is not None and snapshot.revision != expected_revision:
        raise RuntimeError("Session changed before appending")


@dataclass
class PendingAppend:
    """One shared retry identity plus backend-owned native input data."""

    message: SessionMessage
    receipt: asyncio.Future
    payload: Any = None
    submitted: bool = False

    async def wait(self):
        """Cancellation of an observer must not cancel the session's receipt."""
        return (await asyncio.shield(self.receipt)).model_copy(deep=True)


class SessionAppends:
    """Serialize append admission and publication independently from generation."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.pending: dict[str, PendingAppend] = {}

    def get(self, identifier, content, role):
        """Recover an in-flight retry before reading partially committed native state."""
        pending = self.pending.get(identifier)
        if pending is not None:
            check_identity(pending.message, content, role)
        return pending

    def add(self, message, payload=None):
        """Register before dispatch so uncertain writes cannot be duplicated."""
        receipt = asyncio.get_running_loop().create_future()
        receipt.add_done_callback(lambda future: None if future.cancelled() else future.exception())
        pending = PendingAppend(message, receipt, payload)
        self.pending[message.id] = pending
        return pending

    def resolve(self, message):
        """Publish only after the backend's durable state agrees with the input."""
        pending = self.pending.pop(message.id)
        if not pending.receipt.done():
            pending.receipt.set_result(message.model_copy(deep=True))

    def fail(self, error):
        """Release waiters on execution failure; persisted IDs remain recoverable."""
        for pending in self.pending.values():
            if not pending.receipt.done():
                pending.receipt.set_exception(error)
        self.pending.clear()
