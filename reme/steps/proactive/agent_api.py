"""Service Job adapters and request-owned conversation subscriptions."""

import asyncio
from contextlib import aclosing
from time import monotonic

from .base import BaseAgentStep
from .chat.agent import BaseChatAgentStep
from ...components.proactive.outbox import NotificationOutbox
from ...constants import CHAT_AGENT_NAME, PROACTIVE_MESSAGE_ID_PREFIX
from ...core.registry import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk, SessionSnapshot, SessionStatus
from ...schema.agent_inbox import NotificationRef
from ...schema.agent_view import (
    AgentView,
    AgentSessionView,
    AgentSnapshot,
    AgentUpdate,
    AgentHeartbeat,
    ClientActivity,
    AgentRequest,
    AskAgentRequest,
    AgentInboxRequest,
    ListTaskRemindersRequest,
    AgentReminderRequest,
    TaskDocumentRequest,
)
from ...schema.proactive_runtime import validate_agent_name
from ...utils.agent_transcript import TranscriptProjector, project_messages
from ...utils.files.path_lock import get_path_lock


class AgentViews:
    """Compose views from resource owners; subscriptions never own model calls."""

    def __init__(self, step):
        self.step = step
        self.manager = step.agent_manager
        self.chat = step.agent_runtime

    async def runtime(self, name):
        """Resolve Remy separately from Manager-owned background identities."""
        return self.chat if name == CHAT_AGENT_NAME else await self.manager.resolve(name)

    async def view(self, name):
        """Compose a public Agent view from each resource owner."""
        runtime = await self.runtime(name)
        if name == CHAT_AGENT_NAME:
            config, agent_type = {}, "chat"
            running = runtime.busy or runtime.notification_worker is not None
            document = {"memory_path": runtime.to_workspace_relative(runtime.memory_path)}
            facts = {
                "error": runtime.error,
                "last_started_at": runtime.last_started_at.isoformat() if runtime.last_started_at else None,
                "last_completed_at": runtime.last_completed_at.isoformat() if runtime.last_completed_at else None,
                "next_trigger_at": runtime.retry_at.isoformat() if runtime.retry_at else None,
            }
        else:
            facts = await runtime.status()
            config = facts["config"]
            agent_type = config["agent_type"]
            running = facts["running"]
            document = {"task_path": facts["task_path"]}
        session, status, error = None, "idle", facts.get("error")
        if runtime.outbox.issues:
            await runtime.outbox.entries()
            error = "; ".join(filter(None, [error, *runtime.outbox.issues.values()]))[:2000]
        try:
            session = await runtime.current_session()
            if session is not None:
                if await runtime.agent_wrapper.session_exists(session.session_id):
                    status = (await runtime.agent_wrapper.get_session_status(session.session_id)).status
                else:
                    status = "not_loaded"
        except Exception as exc:
            status, error = "error", str(exc)
        if running:
            status = "active"
        elif error:
            status = "error"
        return AgentView(
            agent_name=name,
            agent_type=agent_type,
            enabled=config.get("enabled", getattr(runtime, "enabled", True)),
            schedule=config.get("schedule"),
            min_trigger_interval_seconds=config.get("min_trigger_interval_seconds", 0),
            notify_window=config.get("notify_window"),
            status=status,
            session_id=session.session_id if session else None,
            directory=runtime.to_workspace_relative(runtime.directory),
            pending_events=facts.get("pending_events", 0),
            in_flight_events=facts.get("in_flight_events", 0),
            last_started_at=facts.get("last_started_at"),
            last_completed_at=facts.get("last_completed_at"),
            next_trigger_at=facts.get("next_trigger_at"),
            error=error,
            **document,
        )

    async def list(self):
        """Page persisted references without opening a native session."""
        definitions = await self.manager.discover()
        items = [(await self.view(CHAT_AGENT_NAME)).model_dump(mode="json")]
        names = sorted(set(definitions) | set(self.manager.registry.issues))
        for name in names:
            try:
                items.append((await self.view(name)).model_dump(mode="json"))
            except (OSError, ValueError, RuntimeError) as exc:
                try:
                    validate_agent_name(name)
                    directory = self.manager.files.checked(self.manager.registry.directory / name)
                    items.append(
                        AgentView(
                            agent_name=name,
                            agent_type="event",
                            status="error",
                            error=str(exc),
                            directory=self.manager.to_workspace_relative(directory),
                            task_path=self.manager.to_workspace_relative(directory / "TASK.md"),
                        ).model_dump(mode="json"),
                    )
                except ValueError:
                    self.manager.logger.error(f"Invalid task directory {name!r}: {exc}")
        return {"items": items, "total": len(items)}

    async def native(self, name):
        """Read native history and scope task output to its latest turn."""
        runtime = await self.runtime(name)
        agent = await self.view(name)
        snapshot = None
        if agent.session_id is not None:
            if await runtime.agent_wrapper.session_exists(agent.session_id):
                snapshot = await runtime.agent_wrapper.read_session(agent.session_id)
                if name != CHAT_AGENT_NAME:
                    snapshot = snapshot.model_copy(
                        update={
                            "messages": [
                                message
                                for message in snapshot.messages
                                if snapshot.last_turn_id and message.turn_id == snapshot.last_turn_id
                            ],
                        },
                    )
            else:
                snapshot = SessionSnapshot(
                    session_id=agent.session_id,
                    state=SessionStatus(session_id=agent.session_id, status="not_loaded"),
                )
        return {
            "agent": agent.model_dump(mode="json"),
            "session_id": agent.session_id,
            "snapshot": snapshot.model_dump(mode="json") if snapshot else None,
        }

    def project(self, runtime, value):
        """Merge native history, live output and portable execution activity."""
        current = runtime.live_turns.overlay(runtime.name, value)
        activities = {}
        candidates = [self.chat, *self.manager.runtimes.values()] if runtime.name == CHAT_AGENT_NAME else [runtime]
        for owner in candidates:
            activity = owner.activity
            if activity is None:
                continue
            if owner is not runtime and owner.worker is None and not owner.execution_lock.locked():
                continue
            if (
                owner is runtime
                and activity.session_id != current["session_id"]
                and not (activity.source == "task" and not activity.session_id)
            ):
                continue
            activities[owner.name] = ClientActivity(
                session_id=activity.session_id,
                turn_id=activity.turn_id,
                source=activity.source,
                phase=activity.phase,
                tool=activity.tool,
                started_at=activity.started_at.isoformat() if activity.started_at else None,
            )
        return AgentSessionView(
            agent=current["agent"],
            messages=project_messages(
                current["session_id"] or "",
                (current["snapshot"] or {}).get("messages", []),
                notifications=runtime.name == CHAT_AGENT_NAME,
            ),
            activities=activities,
        ).model_dump(mode="json")

    async def read(self, name):
        """Return an isolated read-only projection."""
        return self.project(await self.runtime(name), await self.native(name))

    async def events(self, name, interval=1, heartbeat=15):
        """Own a cancellable subscription without starting model execution."""
        runtime = await self.runtime(name)
        wake, listeners = asyncio.Event(), set()
        previous, value, agents = {}, {}, None
        next_poll, last_emit, epoch = 0, monotonic(), -1
        try:
            while not runtime.closed and not self.manager.closed:
                wake.clear()
                for owner in [runtime, *self.manager.runtimes.values()]:
                    if owner.live_turns not in listeners:
                        owner.live_turns.listeners.add(wake)
                        listeners.add(owner.live_turns)
                if not value or monotonic() >= next_poll or epoch != runtime.live_turns.epoch:
                    value = await self.native(name)
                    agents = (await self.list())["items"] if name == CHAT_AGENT_NAME else None
                    epoch, next_poll = runtime.live_turns.epoch, monotonic() + interval
                current = {**self.project(runtime, value), "agents": agents}
                if current != previous:
                    old = {message["id"]: message for message in previous.get("messages", [])}
                    ids = [message["id"] for message in current["messages"]]
                    replace = (
                        not previous
                        or previous["agent"]["session_id"] != current["agent"]["session_id"]
                        or list(old) != ids[: len(old)]
                    )
                    if replace:
                        yield AgentSnapshot.model_validate(current).model_dump(mode="json")
                    else:
                        changed = [message for message in current["messages"] if message != old.get(message["id"])]
                        yield AgentUpdate.model_validate({**current, "messages": changed}).model_dump(mode="json")
                    previous, last_emit = current, monotonic()
                elif monotonic() - last_emit >= heartbeat:
                    yield AgentHeartbeat().model_dump(mode="json")
                    last_emit = monotonic()
                try:
                    await asyncio.wait_for(wake.wait(), timeout=max(0, next_poll - monotonic()))
                    await asyncio.sleep(0.025)
                except TimeoutError:
                    pass
        finally:
            for listener in listeners:
                listener.listeners.discard(wake)

    @staticmethod
    def _chat_message_ids(deliveries):
        values = {}
        for delivery in deliveries:
            if delivery.status != "aborted":
                for identifier in delivery.notification_ids:
                    values.setdefault(identifier, []).append(PROACTIVE_MESSAGE_ID_PREFIX + delivery.message_id)
        return values

    async def reminder(self, reference, *, receipt=None, delivery=None, chat_message_ids=None):
        """Join canonical content with sender delivery and receiver processing facts."""
        item = await self.chat.reminder_item(reference)
        if delivery is None:
            delivery = await NotificationOutbox.read_reference(
                self.manager.files,
                self.manager.registry.directory / reference.source_agent,
                reference.id,
                self.chat.name,
            )
        if receipt is None:
            receipt = await self.chat.inbox.processed(reference.id)
        item["read_at"] = receipt.consume_ts.isoformat() if receipt else None
        item["delivered_at"] = (
            delivery.delivered_at.isoformat()
            if delivery and delivery.delivered_at
            else reference.created_at.isoformat()
        )
        if chat_message_ids is None:
            chat_message_ids = self._chat_message_ids(await self.chat.outbox.entries())
        item["chat_message_ids"] = chat_message_ids.get(reference.id, [])
        item.pop("notification_id", None)
        return item

    async def reminders(self, *, agent_name=None, state="all", offset=0, limit=20):
        """Page reminder views without consuming or claiming references."""
        rows, errors = [], {}
        processing = set(self.chat.active_turn.ids) if self.chat.active_turn else set()
        if agent_name is None:
            references = {item.id: item for item in await self.chat.inbox.pending_references()}
            for receipt in await self.chat.inbox.receipts():
                for identifier in receipt.item_ids:
                    reference = await self.chat.inbox.reference(identifier)
                    if reference is not None:
                        references[identifier] = reference
            deliveries = [(reference, None) for reference in references.values()]
        else:
            runtime = await self.manager.resolve(agent_name)
            deliveries = [(row.notification, row) for row in await runtime.outbox.entries()]
        chat_message_ids = self._chat_message_ids(await self.chat.outbox.entries()) if deliveries else {}
        for reference, delivery in deliveries:
            try:
                receipt = await self.chat.inbox.processed(reference.id)
                status = "processed" if receipt else "processing" if reference.id in processing else "pending"
                if state not in {"all", status}:
                    continue
                item = await self.reminder(
                    reference,
                    receipt=receipt,
                    delivery=delivery,
                    chat_message_ids=chat_message_ids,
                )
                rows.append({**item, "state": status})
            except (OSError, ValueError) as exc:
                errors[reference.record_path] = str(exc)
        rows.sort(key=lambda value: (value["delivered_at"], value["id"]), reverse=True)
        if errors:
            self.chat.error = "; ".join(errors.values())[:2000]
        return {"items": rows[offset : offset + limit], "total": len(rows), "offset": offset, "limit": limit}

    async def read_reminder(self, agent_name, identifier):
        """Resolve a source path or delivered Chat message to retained references."""
        if identifier.startswith(PROACTIVE_MESSAGE_ID_PREFIX) and agent_name == CHAT_AGENT_NAME:
            native_id = identifier[len(PROACTIVE_MESSAGE_ID_PREFIX) :]
            chat_deliveries = await self.chat.outbox.entries()
            delivery = next(
                (row for row in chat_deliveries if row.message_id == native_id and row.status != "aborted"),
                None,
            )
            if delivery is None:
                raise ValueError("Chat reminder has no retained source messages")
            references = [await self.chat.inbox.reference(value) for value in delivery.notification_ids]
            if any(reference is None for reference in references):
                raise ValueError("Chat reminder source reference is unavailable")
            chat_message_ids = self._chat_message_ids(chat_deliveries)
            return {
                "id": identifier,
                "items": [
                    await self.reminder(reference, chat_message_ids=chat_message_ids) for reference in references
                ],
            }
        delivery = None
        if agent_name == CHAT_AGENT_NAME:
            reference = await self.chat.inbox.reference(NotificationRef.identity(identifier))
        else:
            runtime = await self.manager.resolve(agent_name)
            delivery = next(
                (row for row in await runtime.outbox.entries() if row.notification.record_path == identifier),
                None,
            )
            reference = delivery.notification if delivery else None
        if reference is None:
            raise ValueError("Message not found in this mailbox")
        return await self.reminder(reference, delivery=delivery)


class _AgentViewStep(BaseAgentStep):
    def request(self, schema=AgentRequest):
        """Validate a facade request before accessing runtime resources."""
        request = schema.model_validate(self.parameters())
        self.require_paths(self.workspace_path)
        return request

    @property
    def views(self):
        """Compose request-scoped service views over the configured owners."""
        return AgentViews(self)


@R.register("agent_list_step")
class AgentListStep(_AgentViewStep):
    """List the independently owned Chat and task Agent views."""

    async def execute(self):
        if self.parameters():
            raise ValueError("list_agents accepts no arguments")
        self.require_paths(self.workspace_path)
        self.context.response.answer = await self.views.list()
        return self.context.response


@R.register("list_chat_sessions_step")
class ListChatSessionsStep(_AgentViewStep):
    """Read Remy session references without initializing a conversation."""

    async def execute(self):
        values = self.parameters()
        if set(values) - {"offset", "limit"}:
            raise ValueError("Session listing accepts only offset and limit")
        self.require_paths(self.workspace_path)
        self.context.response.answer = await self.agent_runtime.sessions.list(**values)
        return self.context.response


@R.register("agent_session_step")
class AgentSessionStep(_AgentViewStep):
    """Read the current public conversation snapshot."""

    async def execute(self):
        request = self.request()
        self.context.response.answer = await self.views.read(request.agent_name)
        return self.context.response


@R.register("task_document_step")
class TaskDocumentStep(_AgentViewStep):
    """Edit the original task document using optimistic file versions."""

    async def execute(self):
        request = self.request(TaskDocumentRequest)
        runtime = await self.agent_manager.resolve(request.agent_name)
        self.require_paths(runtime.task_path)
        path = runtime.to_workspace_relative(runtime.task_path)
        if request.action == "read":
            async with await get_path_lock(runtime.task_path):
                await self.dispatch_steps(["load_step"], path=path)
        else:
            await self.dispatch_steps(
                ["save_step"],
                path=path,
                content=request.content,
                expected_mtime=request.expected_mtime,
            )
        response = self.context.response
        if response.success:
            response.answer = {
                "path": path,
                "mtime": response.metadata["mtime"],
                **({"content": response.answer} if request.action == "read" else {}),
            }
        return response


@R.register("watch_agent_session_step")
class WatchAgentSessionStep(_AgentViewStep):
    """Stream request-owned conversation snapshots and updates."""

    async def execute(self):
        request = self.request()
        async with aclosing(self.views.events(request.agent_name)) as events:
            async for event in events:
                await self.context.add_stream_chunk(StreamChunk(chunk_type=ChunkEnum.DATA, chunk=event))
        return self.context.response


@R.register("ask_agent_step")
class AskAgentStep(_AgentViewStep):
    """Route foreground turns and read-only task questions through their execution owners."""

    async def execute(self):
        request = self.request(AskAgentRequest)
        if not request.query.strip():
            raise ValueError("query must not be blank")
        projector = TranscriptProjector()

        async def publish(chunk):
            projector.update(chunk)
            if projector.error:
                self.context.stream_error = projector.error
            if request.agent_name == CHAT_AGENT_NAME and chunk.chunk_type not in {
                ChunkEnum.REPLY_START,
                ChunkEnum.REPLY_END,
                ChunkEnum.ERROR,
            }:
                return
            await self.context.add_stream_chunk(
                StreamChunk(chunk_type=ChunkEnum.DATA, chunk=projector.message().model_dump(mode="json")),
            )

        if request.agent_name == CHAT_AGENT_NAME:
            response = await self.run_job(
                "run_chat_agent",
                agent_name=request.agent_name,
                query=request.query,
                response=self.context.response,
                _on_chunk=publish,
            )
            response.success = response.success and self.context.stream_error is None
            return response
        reply = await self.query_reply(request.agent_name, request.query, on_chunk=publish)
        self.context.response.success = reply["status"] == "completed" and self.context.stream_error is None
        self.context.response.answer = reply["result"]
        self.context.response.metadata.update(reply, agent_name=request.agent_name)
        return self.context.response


@R.register("agent_new_session_step")
class AgentNewSessionStep(_AgentViewStep, BaseChatAgentStep):
    """Start a fresh Remy conversation using its formal Chat binding."""

    async def execute(self):
        if self.parameters():
            raise ValueError("create_chat_session accepts no arguments")
        self.require_paths(self.workspace_path)
        await self.new_session()
        self.context.response.answer = await self.views.read(CHAT_AGENT_NAME)
        return self.context.response


@R.register("list_task_reminders_step")
class ListTaskRemindersStep(_AgentViewStep):
    """Read reminders produced by one background task."""

    async def execute(self):
        request = self.request(ListTaskRemindersRequest)
        self.context.response.answer = await self.views.reminders(**request.model_dump())
        return self.context.response


@R.register("agent_reminder_step")
class AgentReminderStep(_AgentViewStep):
    """Read source provenance without acknowledging a reminder."""

    async def execute(self):
        request = self.request(AgentReminderRequest)
        self.context.response.answer = await self.views.read_reminder(request.agent_name, request.message_id)
        return self.context.response


@R.register("agent_inbox_step")
class AgentInboxStep(_AgentViewStep):
    """Expose typed input views without consuming a queue."""

    async def execute(self):
        request = self.request(AgentInboxRequest)
        values = request.model_dump()
        name = values.pop("agent_name")
        if name == CHAT_AGENT_NAME:
            answer = await self.views.reminders(**values)
        else:
            runtime = await self.agent_manager.resolve(name)
            answer = await runtime.inbox.read(**values)
            answer["items"] = [
                {"event": row["item"], "state": row["state"], "consume_ts": row["consume_ts"]}
                for row in answer["items"]
            ]
        self.context.response.answer = answer
        return self.context.response
