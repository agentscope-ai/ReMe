"""File-native SSE subscriptions with durable, independently retried delivery."""

import asyncio
from hashlib import sha256
import json
import math
import re
from uuid import uuid4

import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue

from .sse import read_sse
from ...base import BaseComponent
from ..manager import BaseAgentManager
from ....constants import SSE_MEDIA_TYPE
from ....core.registry import R
from ....enumeration import ComponentEnum
from ....schema.cloud_event import CloudEvent
from ....schema.event import Event
from ....schema.proactive_agent import ConnectorSource, validate_runtime_names
from ....utils.async_utils import complete_in_thread, finish_cleanup
from ....utils.files.path_lock import get_path_lock
from ....utils.files.workspace_files import WorkspaceFiles


class PendingDelivery(BaseModel):
    """Freeze recipients and the next cursor before attempting any handoff."""

    model_config = ConfigDict(extra="forbid")
    event: Event
    remaining: list[str]
    cursor: str


class ConnectorCheckpoint(BaseModel):
    """Only committed cursors are used for Last-Event-ID on a new connection."""

    model_config = ConfigDict(extra="forbid")
    cursor: str = ""
    pending: PendingDelivery | None = None


class CloudEventResource(BaseModel):
    """Retain original JSON and a stable fallback time when producers omit time."""

    model_config = ConfigDict(extra="forbid")
    cloud_event: dict[str, JsonValue]
    received_at: AwareDatetime


@R.register("local")
class ConnectorManager(BaseComponent):
    """Own HTTP clients and subscription tasks independently of invocation-scoped Steps."""

    component_type = ComponentEnum.CONNECTOR_MANAGER

    def __init__(self, agent_manager="default", reconnect_seconds=1.0, max_event_bytes=1024 * 1024, **kwargs):
        super().__init__(**kwargs)
        if (
            type(reconnect_seconds) not in (int, float)
            or not math.isfinite(reconnect_seconds)
            or reconnect_seconds <= 0
            or type(max_event_bytes) is not int
            or not 0 < max_event_bytes <= 1024 * 1024
        ):
            raise ValueError("Connector reconnect delay must be positive; event limit must be within 1 MiB")
        self.agent_manager = (
            self.bind(agent_manager, BaseAgentManager, optional=False)
            if isinstance(agent_manager, str)
            else agent_manager
        )
        self.reconnect_seconds = float(reconnect_seconds)
        self.max_event_bytes = max_event_bytes
        self.files = WorkspaceFiles(self.workspace_path)
        self.sources = {}
        self.statuses = {}
        self.tasks = {}
        self.client = None
        self.closed = True
        self.operations = asyncio.Lock()

    @property
    def directory(self):
        """Keep new subscription definitions separate from preset Job controls."""
        metadata = self.app_context.app_config.metadata_dir if self.app_context else "metadata"
        return self.files.path(f"{metadata}/proactive/sources")

    @property
    def resource_root(self):
        """Store one immutable source record per CloudEvents source/id pair."""
        resource = self.app_context.app_config.resource_dir if self.app_context else "resource"
        root = self.files.path(resource)
        if root == self.files.workspace:
            raise ValueError("resource_dir must be a workspace subdirectory")
        return self.files.checked(root / "cloudevents")

    def source_directory(self, source_id):
        """Validate identities before using them in file paths, including direct component calls."""
        if not isinstance(source_id, str) or not re.fullmatch(r"source_[0-9a-f]{32}", source_id):
            raise ValueError("Invalid Connector source_id")
        return self.files.checked(self.directory / source_id)

    def _load_source(self, source_id):
        path = self.source_directory(source_id) / "config.json"
        source = ConnectorSource.model_validate(self.files.read_json(path, None))
        if source.source_id != source_id:
            raise ValueError("Connector source identity conflicts with its directory")
        return source

    def _load_sources(self):
        if not self.directory.exists():
            return {}
        sources = {}
        for path in sorted(self.directory.glob("*/config.json")):
            source = self._load_source(path.parent.name)
            sources[source.source_id] = source
        return sources

    async def _start(self):
        self.files.checked(self.resource_root)
        self.sources = await asyncio.to_thread(self._load_sources)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=90), follow_redirects=True)
        self.closed = False
        for source in self.sources.values():
            self._spawn(source)

    async def _close(self):
        async with self.operations:
            self.closed = True
            for task in self.tasks.values():
                task.cancel()
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
            self.tasks.clear()
            if self.client is not None:
                await self.client.aclose()
                self.client = None

    def _check_open(self):
        if self.closed:
            raise RuntimeError("ConnectorManager is closed")

    def _spawn(self, source):
        self.statuses.setdefault(source.source_id, {"connected": False, "last_error": None})
        self.statuses[source.source_id]["connected"] = False
        if source.enabled:
            self.tasks[source.source_id] = asyncio.create_task(self._run(source), name=f"connector:{source.source_id}")

    async def _stop(self, source_id):
        task = self.tasks.pop(source_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _validate_targets(self, names):
        for name in names:
            if (await self.agent_manager.definition(name)).agent_type != "event":
                raise ValueError("Connector targets must be event-triggered Runtimes")

    async def _apply_source(self, source, *, original=None):
        """Publish a definition and reconcile its worker without reconnecting for target changes."""
        path = self.source_directory(source.source_id) / "config.json"
        async with await get_path_lock(path):
            if original is not None and await asyncio.to_thread(self._load_source, source.source_id) != original:
                raise ValueError("Connector definition changed on disk; restart to load the edited source")
            if source != original:
                await complete_in_thread(self.files.write_json, path, source.model_dump(mode="json"))
            self.sources[source.source_id] = source
            task = self.tasks.get(source.source_id)
            if not source.enabled:
                await self._stop(source.source_id)
            elif task is None or task.done():
                self._spawn(source)

    async def create(self, values):
        """Persist the definition before starting an enabled subscription."""
        source = ConnectorSource(source_id=f"source_{uuid4().hex}", **values)
        validate_runtime_names(source.runtime_names)
        async with self.operations:
            self._check_open()
            await self._validate_targets(source.runtime_names)

            await finish_cleanup(self._apply_source(source))
            return await self.status(source.source_id)

    async def update(self, source_id, **changes):
        """Replace recipients or pause/resume; saved deliveries keep their original recipients."""
        if not changes or changes.keys() - {"runtime_names", "enabled"}:
            raise ValueError("Connector updates accept runtime_names or enabled")
        async with self.operations:
            self._check_open()
            self.source_directory(source_id)
            if source_id not in self.sources:
                raise ValueError(f"Connector source not found: {source_id}")
            original = self.sources[source_id]
            source = ConnectorSource.model_validate({**original.model_dump(), **changes})
            if "runtime_names" in changes:
                await self._validate_targets(source.runtime_names)

            await finish_cleanup(self._apply_source(source, original=original))
            return await self.status(source_id)

    async def status(self, source_id):
        """Report saved configuration separately from connection and pending handoff state."""
        source = self.sources[source_id]
        state = await self._read_state(source_id)
        return {
            **source.model_dump(mode="json"),
            "available": not self.closed,
            **self.statuses.get(source_id, {"connected": False, "last_error": None}),
            "last_event_id": state.cursor,
            "pending_delivery_count": len(state.pending.remaining) if state.pending else 0,
        }

    async def list_sources(self):
        """List subscriptions without starting collectors or changing their files."""
        return [await self.status(source_id) for source_id in list(self.sources)]

    async def _read_state(self, source_id):
        path = self.source_directory(source_id) / "state.json"
        value = await asyncio.to_thread(self.files.read_json, path, {})
        state = ConnectorCheckpoint.model_validate(value)
        if state.pending:
            validate_runtime_names(state.pending.remaining, allow_empty=True)
            for relative in state.pending.event.paths:
                self.files.path(relative)
        return state

    async def _save_state(self, source_id, state):
        await complete_in_thread(
            self.files.write_json,
            self.source_directory(source_id) / "state.json",
            state.model_dump(mode="json"),
        )
        return state

    async def _resource_event(self, raw):
        cloud = CloudEvent.model_validate(raw)
        identity = json.dumps([cloud.source, cloud.id], ensure_ascii=False, separators=(",", ":"))
        digest = sha256(identity.encode()).hexdigest()
        path = self.files.checked(self.resource_root / f"{digest}.json")
        async with await get_path_lock(path):
            saved = await asyncio.to_thread(self.files.read_json, path, None)
            if saved is None:
                record = CloudEventResource(cloud_event=raw, received_at=self.agent_manager.now())
                await complete_in_thread(self.files.write_json, path, record.model_dump(mode="json"))
            else:
                record = CloudEventResource.model_validate(saved)
                original = CloudEvent.model_validate(record.cloud_event)
                if original.canonical() != cloud.canonical():
                    raise ValueError("Conflicting CloudEvent content for the same source and id")
                cloud = original
        return Event(
            id=f"cloudevent-{digest}",
            event_ts=cloud.time or record.received_at,
            source="cloudevents",
            description=cloud.type + (f": {cloud.subject}" if cloud.subject else ""),
            paths=[self.to_workspace_relative(path)],
            metadata={"cloudevents": {"id": cloud.id, "source": cloud.source, "type": cloud.type}},
        )

    async def _deliver_pending(self, source, state):
        pending = state.pending
        if pending is None:
            return state
        for relative in pending.event.paths:
            await asyncio.to_thread(self.files.read, self.files.path(relative))
        result = await self.agent_manager.deliver_events(
            pending.remaining,
            [pending.event],
        )
        remaining = [name for name in pending.remaining if name not in result["deliveries"]]
        if remaining:
            state = state.model_copy(update={"pending": pending.model_copy(update={"remaining": remaining})})
            await self._save_state(source.source_id, state)
            raise RuntimeError(f"Event delivery blocked: {result['rejected']} {result['errors']}")
        state = ConnectorCheckpoint(cursor=pending.cursor)
        return await self._save_state(source.source_id, state)

    async def _receive(self, source, state, frame):
        source = self.sources[source.source_id]
        cursor = frame.cursor if frame.cursor is not None else state.cursor
        if frame.data is None:
            return await self._save_state(source.source_id, ConnectorCheckpoint(cursor=cursor))
        event = await self._resource_event(json.loads(frame.data))
        pending = PendingDelivery(event=event, remaining=list(source.runtime_names), cursor=cursor)
        state = await self._save_state(source.source_id, state.model_copy(update={"pending": pending}))
        state = await self._deliver_pending(source, state)
        self.statuses[source.source_id]["last_error"] = None
        return state

    async def _run(self, source):
        delay = self.reconnect_seconds
        status = self.statuses[source.source_id]
        while not self.closed:
            try:
                state = await self._deliver_pending(source, await self._read_state(source.source_id))
                headers = {b"Accept": SSE_MEDIA_TYPE.encode(), b"Cache-Control": b"no-cache"}
                if state.cursor:
                    headers[b"Last-Event-ID"] = state.cursor.encode("utf-8")
                async with self.client.stream("GET", str(source.binding.url), headers=headers) as response:
                    if response.status_code == 204:
                        return
                    response.raise_for_status()
                    if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != SSE_MEDIA_TYPE:
                        raise ValueError("Connector endpoint must return text/event-stream")
                    status["connected"] = True
                    status["last_error"] = None
                    async for frame in read_sse(response, self.max_event_bytes):
                        state = await self._receive(source, state, frame)
                        if frame.retry_ms is not None:
                            delay = min(60, max(0.1, frame.retry_ms / 1000))
            except Exception as exc:
                status["last_error"] = (str(exc) or type(exc).__name__)[:2000]
                self.logger.warning(f"Connector {source.source_id}: {status['last_error']}")
            finally:
                status["connected"] = False
            await asyncio.sleep(delay)
