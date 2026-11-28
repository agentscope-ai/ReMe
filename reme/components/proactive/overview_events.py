"""Manager-owned bounded invalidation journal; durable owners remain authoritative."""

from collections import OrderedDict, deque
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4
from time import monotonic

from ...schema.proactive_overview import OverviewTransition
from ...schema.proactive_trace import SourceRef

_METADATA_KEY = "reme.proactive.overview"


def source_ref(event, agent_name):
    """Reuse Event.source; time source identity also includes its owning Runtime."""
    source = event["source"]
    display = event.get("metadata", {}).get("source_ref", {})
    if not isinstance(display, dict):
        display = {}
    display = {key: value for key, value in display.items() if isinstance(value, str) and value}
    return SourceRef(
        source_id=display.get("source_id") or (f"time:{agent_name}" if source == "time" else source),
        name=display.get("name") or source,
        trigger_type="time" if source == "time" else "event",
    )


def notify_overview(owner, kind=None, *, identity=None, **values):
    """Publish synchronously after success, without owning tasks or doing I/O."""
    context = owner.app_context
    journal = context.metadata.get(_METADATA_KEY) if context is not None else None
    if journal is not None:
        journal.publish(kind, identity=identity, **values)


class OverviewEvents:
    """Slow readers resnapshot instead of accumulating an unbounded action backlog."""

    def __init__(self, capacity=512):
        self.capacity = capacity
        self.listeners = set()
        self.projections = OrderedDict()
        self.projection_lock = asyncio.Lock()
        self.closed = True
        self.reset()

    def reset(self):
        """Each lifecycle has an independent version namespace."""
        self.epoch = uuid4().hex
        self.revision = 0
        self.buffer = deque(maxlen=self.capacity)
        self.identities = OrderedDict()
        self.projections.clear()
        self.closed = False

    async def project(self, key, token, read, *, force=False, refresh=5):
        """Share bounded read views; owner changes invalidate immediately, external files refresh periodically."""
        async with self.projection_lock:
            cached = self.projections.get(key)
            if force or cached is None or cached[0] != token or monotonic() >= cached[1]:
                value = await read()
                self.projections[key] = (token, monotonic() + refresh, value)
                if len(self.projections) > 8:
                    self.projections.popitem(last=False)
            self.projections.move_to_end(key)
            return deepcopy(self.projections[key][2])

    def attach(self, context):
        """Application context borrows this journal; the Manager owns its lifetime."""
        if context is not None:
            context.metadata[_METADATA_KEY] = self

    def publish(self, kind=None, *, identity=None, **values):
        """Duplicate successful handoffs keep their original action identity."""
        if self.closed:
            return
        transition = None
        if kind is not None:
            key = f"{kind}:{identity}"
            if key in self.identities:
                return
            transition = OverviewTransition(
                id=sha256(key.encode()).hexdigest(),
                type=kind,
                occurred_at=values.pop("occurred_at", None) or datetime.now(timezone.utc).isoformat(),
                **values,
            ).model_dump(mode="json")
            self.identities[key] = None
            if len(self.identities) > self.capacity * 4:
                self.identities.popitem(last=False)
        self.revision += 1
        self.buffer.append((self.revision, transition))
        for listener in self.listeners:
            listener.set()

    def since(self, revision, *, through=None):
        """None means a version gap requires an authoritative snapshot."""
        if revision > self.revision or (self.buffer and revision < self.buffer[0][0] - 1):
            return None
        return [
            value
            for version, value in self.buffer
            if revision < version <= (self.revision if through is None else through) and value is not None
        ]

    def close(self):
        """Wake request-owned readers so application shutdown can finish."""
        self.closed = True
        self.projections.clear()
        for listener in self.listeners:
            listener.set()
        self.buffer.clear()
        self.identities.clear()
