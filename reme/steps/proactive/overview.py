"""Compose a read-only whole-flow subscription from the original resource owners."""

import asyncio
from contextlib import aclosing
from datetime import datetime
from time import monotonic

from .agent_api import AgentViews
from .base import BaseAgentStep
from ...constants import CHAT_AGENT_NAME, PROACTIVE_MESSAGE_ID_PREFIX
from ...core.registry import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk
from ...schema.agent_inbox import NotificationRef
from ...schema.proactive_connector import ConnectorView
from ...schema.proactive_trace import merge_traces
from ...components.proactive.connector import ConnectorManager
from ..base_step import Ref
from ...enumeration import ComponentEnum
from ...components.proactive.outbox import NotificationOutbox
from ...schema.proactive_overview import (
    WatchProactiveOverviewRequest,
    ProactiveOverview,
    OverviewAgent,
    OverviewRun,
    OverviewInbox,
    OverviewInboxItem,
    OverviewQueueCounts,
    OverviewReminder,
    OverviewFile,
    OverviewSnapshot,
    OverviewUpdate,
    OverviewHeartbeat,
)

_RECENT_ITEMS = 3


def _entities(overview):
    """Stable business identities keep deletions and replacements explicit."""
    result = {}
    for collection in ("sources", "agents", "reminders", "workspace"):
        key = "source_id" if collection == "sources" else "agent_name" if collection == "agents" else "id"
        result[collection] = {item[key]: item for item in overview[collection]}
    for collection in ("event_inbox", "remy_inbox"):
        result[collection] = {collection: overview[collection]}
    result["errors"] = overview["errors"]
    return result


def _changes(previous, current):
    old, new = _entities(previous), _entities(current)
    return [
        {"collection": collection, "id": key, "value": entities.get(key)}
        for collection, entities in new.items()
        for key in sorted(old[collection].keys() | entities.keys())
        if old[collection].get(key) != entities.get(key)
    ]


class ProactiveOverviewViews:
    """No execution, input consumption, native session creation or user receipts."""

    def __init__(self, step):
        self.step = step
        self.manager = step.agent_manager
        self.chat = step.agent_runtime
        self.views = AgentViews(step)
        self.journal = self.manager.overview_events

    async def sources(self, errors):
        """Use the same source catalog as manage_connector."""
        try:
            items = await self.step.connector_manager.list_sources()
            errors.update({f"source:{item['source_id']}": item["last_error"] for item in items if item["last_error"]})
            return {item["source_id"]: ConnectorView.model_validate(item) for item in items}
        except (OSError, ValueError, RuntimeError) as exc:
            errors["sources"] = str(exc)[:2000]
            return {}

    @staticmethod
    def _counts(queue, name, snapshot):
        counts = OverviewQueueCounts(
            **{key: snapshot[key] for key in ("waiting_count", "processing_count", "processed_count")},
        )
        queue.queues[name] = counts
        for key in ("waiting_count", "processing_count", "processed_count"):
            setattr(queue, key, getattr(queue, key) + getattr(counts, key))

    @staticmethod
    def _last_run(runtime, receipt):
        value = runtime.last_run
        if value is None and receipt is not None and receipt.metadata.get("run_id"):
            value = {
                **{
                    key: receipt.metadata[key]
                    for key in (
                        "run_id",
                        "event_ids",
                        "source_refs",
                        "record_paths",
                        "session_id",
                        "started_at",
                        "notification_requested",
                    )
                    if key in receipt.metadata
                },
                "phase": "completed",
                "completed_at": receipt.consume_ts.isoformat(),
            }
        return OverviewRun.model_validate(value) if value else None

    async def read(self, names=None):
        """Assemble six bounded sections, isolating damaged owners in errors."""
        definitions = await self.manager.discover()
        if names is not None:
            missing = set(names) - set(definitions)
            if missing:
                raise ValueError(f"Unknown background Agents: {', '.join(sorted(missing))}")
        selected = sorted(set(definitions) | set(self.manager.registry.issues)) if names is None else names
        errors, agents, workspace, references, deliveries = {}, [], {}, {}, {}
        sources = await self.sources(errors)
        configured_sources = set(sources)
        if names is not None:
            sources = {key: value for key, value in sources.items() if set(value.runtime_names).intersection(names)}
        event_inbox, remy_inbox = OverviewInbox(), OverviewInbox()

        def add_file(path, role, agent_name=None, run_id=None):
            try:
                absolute = self.manager.files.path(path)
                self.step.require_paths(absolute)
                stamp = datetime.fromtimestamp(absolute.stat().st_mtime).isoformat() if absolute.exists() else None
                identifier = f"{agent_name or ''}:{role}:{path}"
                workspace[identifier] = OverviewFile(
                    id=identifier,
                    agent_name=agent_name,
                    role=role,
                    path=path,
                    mtime=stamp,
                    run_id=run_id,
                )
            except (OSError, ValueError) as exc:
                errors[f"file:{path}"] = str(exc)[:2000]

        config = self.step.app_context.app_config
        add_file(config.resource_dir, "resource")
        add_file(config.daily_dir, "record")
        add_file(self.chat.to_workspace_relative(self.chat.memory_path), "memory", CHAT_AGENT_NAME)
        for name in selected:
            try:
                runtime = await self.manager.resolve(name)
                view = await self.views.view(name)
                snapshot = await runtime.inbox.overview(limit=_RECENT_ITEMS)
                entries = await runtime.outbox.entries()
                for identifier, error in runtime.outbox.issues.items():
                    errors[f"delivery:{name}:{identifier}"] = error[:2000]
                recent_deliveries = sorted(entries, key=lambda value: value.created_at, reverse=True)[:_RECENT_ITEMS]
                for delivery in recent_deliveries:
                    deliveries[delivery.notification.id] = delivery
                    references[delivery.notification.id] = delivery.notification
                    add_file(
                        delivery.notification.record_path,
                        "record",
                        name,
                        delivery.notification.trace.run_id if delivery.notification.trace else None,
                    )
                active = runtime.active_run
                if active is not None:
                    active = {
                        **active,
                        "phase": runtime.activity.phase if runtime.activity else active["phase"],
                        "session_id": runtime.activity.session_id if runtime.activity else None,
                        "turn_id": runtime.activity.turn_id if runtime.activity else active["run_id"],
                        "tool": runtime.activity.tool if runtime.activity else None,
                    }
                    if active["phase"] == "completed":
                        active["phase"] = "committing"
                agent = OverviewAgent(
                    **view.model_dump(mode="json"),
                    active_run=active,
                    last_run=self._last_run(runtime, snapshot["receipt"]),
                    deliveries=recent_deliveries,
                    pending_delivery_count=sum(item.status == "prepared" for item in entries),
                    delivered_count=sum(item.status == "delivered" for item in entries),
                )
                agents.append(agent)
                add_file(view.task_path, "task", name)
                for run in (agent.active_run, agent.last_run):
                    if run:
                        for path in run.record_paths:
                            add_file(path, "record", name, run.run_id)
                self._counts(event_inbox, name, snapshot)
                for row in snapshot["items"]:
                    item = row["item"]
                    event_inbox.items.append(
                        OverviewInboxItem(
                            id=item["id"],
                            agent_name=name,
                            event=item,
                            state=row["state"],
                            received_at=row["received_at"],
                            consume_ts=row["consume_ts"],
                            source_refs=[row["source_ref"]],
                            event_ids=[item["id"]],
                            run_ids=[row["run_id"]] if row["run_id"] else [],
                            record_paths=row["metadata"].get("record_paths", []),
                        ),
                    )
                    for path in item["paths"]:
                        add_file(path, "resource", name, row["run_id"])
                for source in snapshot["sources"]:
                    identifier = source["source_id"]
                    if identifier in configured_sources and identifier not in sources:
                        continue
                    value = sources.get(identifier)
                    if value is None:
                        value = sources[identifier] = ConnectorView(**source)
                    if identifier not in configured_sources and name not in value.runtime_names:
                        value.runtime_names.append(name)
                    value.last_admitted_at = max(
                        filter(None, [value.last_admitted_at, source["last_admitted_at"]]),
                        key=datetime.fromisoformat,
                        default=None,
                    )
                if view.agent_type == "time":
                    identifier = f"time:{name}"
                    previous = sources.get(identifier)
                    sources[identifier] = ConnectorView(
                        source_id=identifier,
                        name=name,
                        trigger_type="time",
                        runtime_names=[name],
                        enabled=view.enabled,
                        available=True,
                        schedule=view.schedule,
                        next_trigger_at=view.next_trigger_at,
                        last_admitted_at=previous.last_admitted_at if previous else None,
                    )
            except (OSError, ValueError, RuntimeError) as exc:
                errors[f"agent:{name}"] = str(exc)[:2000]
                if not any(agent.agent_name == name for agent in agents):
                    try:
                        directory = self.manager.files.checked(self.manager.registry.directory / name)
                        agents.append(
                            OverviewAgent(
                                agent_name=name,
                                agent_type=definitions[name].agent_type if name in definitions else "event",
                                directory=self.manager.to_workspace_relative(directory),
                                task_path=self.manager.to_workspace_relative(directory / "TASK.md"),
                                status="error",
                                error=str(exc)[:2000],
                            ),
                        )
                    except ValueError:
                        pass
        errors.update(
            {
                f"agent:{name}": error[:2000]
                for name, error in self.manager.registry.issues.items()
                if names is None or name in names
            },
        )
        agents.append(OverviewAgent(**(await self.views.view(CHAT_AGENT_NAME)).model_dump(mode="json")))
        chat_deliveries = await self.chat.outbox.entries()
        errors.update({f"chat_delivery:{key}": value[:2000] for key, value in self.chat.outbox.issues.items()})
        snapshot = await self.chat.inbox.overview(limit=_RECENT_ITEMS, source_agents=names)
        self._counts(remy_inbox, CHAT_AGENT_NAME, snapshot)
        for row in snapshot["items"]:
            reference = NotificationRef.model_validate(row["item"])
            references[reference.id] = reference
            if reference.id not in deliveries:
                try:
                    deliveries[reference.id] = await NotificationOutbox.read_reference(
                        self.manager.files,
                        self.manager.registry.directory / reference.source_agent,
                        reference.id,
                        self.chat.name,
                    )
                except (OSError, ValueError) as exc:
                    errors[f"delivery:{reference.id}"] = str(exc)[:2000]
            add_file(
                reference.record_path,
                "record",
                reference.source_agent,
                reference.trace.run_id if reference.trace else None,
            )
            remy_inbox.items.append(
                OverviewInboxItem(
                    id=reference.id,
                    agent_name=CHAT_AGENT_NAME,
                    notification=reference,
                    state=row["state"],
                    received_at=row["received_at"],
                    consume_ts=row["consume_ts"],
                    record_paths=[reference.record_path],
                    delivery=deliveries.get(reference.id),
                    **merge_traces([reference.trace]),
                    chat_message_ids=[
                        PROACTIVE_MESSAGE_ID_PREFIX + value.message_id
                        for value in chat_deliveries
                        if value.status in {"sent", "acknowledged"} and reference.id in value.notification_ids
                    ],
                ),
            )
        reminders = []
        for delivery in sorted(
            chat_deliveries,
            key=lambda value: value.delivered_at.timestamp() if value.delivered_at else -float("inf"),
            reverse=True,
        ):
            if delivery.status not in {"sent", "acknowledged"}:
                continue
            linked = []
            for identifier in delivery.notification_ids:
                reference = references.get(identifier) or await self.chat.inbox.reference(identifier)
                if reference is not None:
                    linked.append(reference)
            if names is not None and not any(ref.source_agent in names for ref in linked):
                continue
            for reference in linked:
                add_file(
                    reference.record_path,
                    "record",
                    reference.source_agent,
                    reference.trace.run_id if reference.trace else None,
                )
            reminders.append(
                OverviewReminder(
                    id=PROACTIVE_MESSAGE_ID_PREFIX + delivery.message_id,
                    session_id=delivery.session_id,
                    summary=delivery.content[:500],
                    status=delivery.status,
                    delivered_at=delivery.delivered_at.isoformat(),
                    notification_ids=delivery.notification_ids,
                    record_paths=list(delivery.record_versions),
                    **merge_traces(ref.trace for ref in linked),
                ),
            )
            if len(reminders) >= _RECENT_ITEMS:
                break
        for queue in (event_inbox, remy_inbox):
            queue.items.sort(
                key=lambda item: (
                    datetime.fromisoformat(item.received_at).timestamp() if item.received_at else -float("inf"),
                    item.agent_name,
                    item.id,
                ),
                reverse=True,
            )
            queue.items = queue.items[:_RECENT_ITEMS]
        return ProactiveOverview(
            sources=sorted(sources.values(), key=lambda value: value.source_id),
            agents=agents,
            event_inbox=event_inbox,
            remy_inbox=remy_inbox,
            reminders=reminders,
            workspace=sorted(workspace.values(), key=lambda value: value.id),
            errors=errors,
        ).model_dump(mode="json")

    async def events(self, names=None, *, interval=1, heartbeat=15):
        """Connections always start with a snapshot; recovered actions never replay."""
        journal = self.journal
        wake, listeners = asyncio.Event(), set()
        previous, revision, epoch = None, 0, journal.epoch
        last_emit = monotonic()
        journal.listeners.add(wake)
        try:
            while not journal.closed and not self.manager.closed and not self.chat.closed:
                wake.clear()
                for runtime in [self.chat, *self.manager.runtimes.values()]:
                    if runtime.live_turns not in listeners:
                        runtime.live_turns.listeners.add(wake)
                        listeners.add(runtime.live_turns)
                before = journal.revision
                observed_epoch = journal.epoch
                activity = tuple(
                    (
                        runtime.name,
                        runtime.activity.model_dump_json() if runtime.activity else None,
                        bool(runtime.worker),
                        runtime.execution_lock.locked(),
                        runtime.error,
                        getattr(runtime, "retry_at", None),
                        getattr(runtime, "delivery_error", None),
                    )
                    for runtime in [self.chat, *self.manager.runtimes.values()]
                )
                current = await journal.project(
                    tuple(sorted(names)) if names is not None else None,
                    (
                        observed_epoch,
                        before,
                        activity,
                        tuple(
                            (key, tuple(sorted(value.items())))
                            for key, value in self.step.connector_manager.statuses.items()
                        ),
                    ),
                    lambda: self.read(names),
                    force=previous is None,
                )
                if journal.epoch != observed_epoch:
                    continue
                if journal.closed:
                    break
                # Anchor to the start of this read. Concurrent changes stay in the journal
                # for the next update, even when producers never become idle.
                transitions = journal.since(revision, through=before)
                reset = previous is None or epoch != journal.epoch or transitions is None
                if reset:
                    revision, epoch = before, observed_epoch
                    event = OverviewSnapshot(
                        epoch=epoch,
                        revision=revision,
                        as_of=self.manager.now().isoformat(),
                        overview=current,
                    )
                else:
                    changes = _changes(previous, current)
                    if names is not None:
                        transitions = [
                            value
                            for value in transitions
                            if value["agent_name"] in names
                            or value["agent_name"] == CHAT_AGENT_NAME
                            and any(
                                path in value["paths"] for item in current["reminders"] for path in item["record_paths"]
                            )
                        ]
                    if changes or transitions:
                        if before == revision:
                            journal.publish()
                            continue
                        event = OverviewUpdate(
                            epoch=epoch,
                            previous_revision=revision,
                            revision=before,
                            as_of=self.manager.now().isoformat(),
                            changes=changes,
                            transitions=transitions,
                        )
                        revision = before
                    elif monotonic() - last_emit >= heartbeat:
                        event = OverviewHeartbeat()
                    else:
                        event = None
                previous = current
                if event is not None:
                    last_emit = monotonic()
                    yield event.model_dump(mode="json")
                try:
                    await asyncio.wait_for(wake.wait(), timeout=min(interval, heartbeat))
                    await asyncio.sleep(0.025)
                except TimeoutError:
                    pass
        finally:
            journal.listeners.discard(wake)
            for listener in listeners:
                listener.listeners.discard(wake)


@R.register("watch_proactive_overview_step")
class WatchProactiveOverviewStep(BaseAgentStep):
    """The only new public Job; detailed history and file content use existing Jobs."""

    connector_manager = Ref(ConnectorManager, ComponentEnum.CONNECTOR_MANAGER)

    async def execute(self):
        request = WatchProactiveOverviewRequest.model_validate(self.parameters())
        self.require_paths(self.workspace_path)
        views = ProactiveOverviewViews(self)
        async with aclosing(views.events(request.agent_names)) as events:
            async for event in events:
                await self.context.add_stream_chunk(StreamChunk(chunk_type=ChunkEnum.DATA, chunk=event))
        return self.context.response
