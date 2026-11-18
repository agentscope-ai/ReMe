"""Application-owned agent coordination with file-native recovery."""

import asyncio
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
import math
import shutil
from zoneinfo import ZoneInfo

from watchfiles import awatch

from .base_agent_runtime import BaseAgentRuntime
from ._internal.files import AgentFiles
from ._internal.notifications import ChatNotifications
from ._internal.outbox import AgentOutbox
from ._internal.registry import AgentRegistry
from ._internal.sessions import AgentSessions
from ._internal.live_turn import LiveTurns
from ..agent_wrapper import BaseAgentWrapper
from ..agent_inbox import BaseAgentInbox
from ..component_registry import R
from ...schema.agent_runtime import (
    ChatAgentConfig,
    CreateChatAgentConfig,
    NotificationConfig,
)
from ...schema.agent_activity import AgentActivity
from ...schema.application_config import ApplicationConfig
from ...schema.event import Event
from ...schema.proactive_agent import CreateProactiveAgentConfig
from ...utils.async_utils import complete_in_thread, finish_cleanup


@dataclass
class _Execution:
    started_at: datetime | None = None
    error: str | None = None
    retry_at: datetime | None = None
    retry_items: list[str] | None = None


@R.register("local")
class LocalAgentRuntime(BaseAgentRuntime):
    """Own task lifetimes and rebuild execution facts from their source logs."""

    def __init__(
        self,
        agent_wrapper="default",
        inbox="default",
        notifications=None,
        retry_seconds=5,
        agents=None,
        chat=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if type(retry_seconds) not in (int, float) or not math.isfinite(retry_seconds) or retry_seconds < 0:
            raise ValueError("retry_seconds must be a finite nonnegative number")
        if agents is not None and not isinstance(agents, list):
            raise ValueError("agents must be a list of initial task configurations")
        self.initial_agents = [CreateProactiveAgentConfig.model_validate(value) for value in agents or []]
        if len({agent.agent_name.casefold() for agent in self.initial_agents}) != len(self.initial_agents):
            raise ValueError("initial Agent names must be unique, including case")
        self.initial_chat = CreateChatAgentConfig.model_validate(chat if chat is not None else {})
        self.agent_wrapper = self.bind(agent_wrapper, BaseAgentWrapper, optional=False)
        self.inbox = self.bind(inbox, BaseAgentInbox, optional=False)
        self.files = AgentFiles(self.workspace_path)
        directory = (
            self.app_context.app_config.reme_agent_dir if self.app_context else ApplicationConfig().reme_agent_dir
        )
        self.agents = AgentRegistry(self.files, directory)
        self.sessions = AgentSessions(self.agents)
        self.live_turns = LiveTurns()
        self.outbox = AgentOutbox(self)
        self.notifications = ChatNotifications(
            self,
            NotificationConfig.model_validate(notifications if notifications is not None else {}),
        )
        self.retry_seconds = retry_seconds
        self.chat_lock = asyncio.Lock()
        self.chat_tasks = set()
        self.workers = {}
        self.agent_tasks = {}
        self.activities = {}
        self.executions = {}
        self.cron = {}
        self.changed = asyncio.Event()
        self.round_locks = {}
        self.management_lock = asyncio.Lock()
        self.watch_stop = asyncio.Event()
        self.watcher = None
        self.closed = True

    def now(self):
        """Use the application timezone for dates and trigger calculation."""
        timezone = self.app_context.app_config.timezone if self.app_context else None
        return datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()

    async def _start(self):
        if self.app_context is None:
            raise RuntimeError("LocalAgentRuntime requires an ApplicationContext")
        chat = self.agents.path("Remy") / "config.json"
        config = self.files.read_json(chat, None)
        config = (
            ChatAgentConfig.model_validate(config) if config is not None else ChatAgentConfig(created_at=self.now())
        )
        await complete_in_thread(
            self.files.write,
            self.agents.chat_memory_path(),
            self.initial_chat.initial_memory,
            create_only=True,
        )
        await complete_in_thread(self.files.write_json, chat, config.model_dump(mode="json"))
        existing = {name.casefold() for name in await asyncio.to_thread(self.agents.load, include_deleted=True)}
        for initial in self.initial_agents:
            if initial.agent_name.casefold() not in existing:
                await self.create(initial)
        for agent in (await asyncio.to_thread(self.agents.load)).values():
            await self.prepare(agent)
        await self.inbox.mount("Remy", (chat.parent / "inbox").relative_to(self.workspace_path).as_posix())
        await self.outbox.sync()
        self.notifications.reset()
        self.closed = False
        self.changed.set()
        self.watch_stop.clear()
        self.watcher = asyncio.create_task(self._watch(), name="agents:watch")

    async def _close(self):
        self.closed = True
        self.live_turns.clear()
        self.watch_stop.set()
        self.changed.set()
        await self.notifications.close()
        tasks = set(self.workers.values()) | set(self.agent_tasks) | self.chat_tasks
        if self.watcher is not None:
            tasks.add(self.watcher)
        for task in tasks:
            task.cancel()
        if tasks:
            await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
        self.workers.clear()
        self.agent_tasks.clear()
        self.activities.clear()
        self.chat_tasks.clear()
        self.executions.clear()
        self.cron.clear()
        self.round_locks.clear()
        self.watcher = None

    async def _watch(self):
        """One watcher plus timeout rescans covers edits, replacement and missed notifications."""
        async with aclosing(
            awatch(
                self.agents.directory,
                stop_event=self.watch_stop,
                yield_on_timeout=True,
                rust_timeout=1000,
                debounce=100,
                step=50,
            ),
        ) as watcher:
            async for _ in watcher:
                if self.closed:
                    break
                try:
                    await self.outbox.sync()
                except (OSError, ValueError) as exc:
                    self.logger.error(f"Could not reconcile agent files: {exc}")
                self.changed.set()

    async def prepare(self, agent):
        """Mount one per-agent inbox without creating independent component lifetimes."""
        directory = (self.agents.directory_for(agent) / "inbox").relative_to(self.workspace_path).as_posix()
        await self.inbox.mount(agent.agent_name, directory)
        self.executions.setdefault(agent.agent_name, _Execution())

    async def definition(self, agent_name):
        """Read current user-owned configuration rather than a stale runtime copy."""
        return await asyncio.to_thread(self.agents.get, agent_name)

    async def create(self, config):
        """Persist a task and its inbox directory before returning success."""
        async with self.management_lock:
            agent, restored = await complete_in_thread(
                self.agents.create,
                config,
                self.now(),
            )
            await self.prepare(agent)
            self.changed.set()
            return {**await self._status(agent), "restored": restored}

    async def update(self, agent_name, config):
        """Apply execution settings to future rounds and invalidate the in-memory timer."""
        async with self.management_lock:
            agent = await complete_in_thread(self.agents.update, agent_name, config)
            self.cron.pop(agent_name, None)
            await self.prepare(agent)
            self.changed.set()
            return await self._status(agent)

    async def delete(self, agent_name):
        """Stop owned work while retaining a restorable identity and all source files."""
        async with self.management_lock:
            tasks = {task for task, owner in self.agent_tasks.items() if owner == agent_name}
            if agent_name in self.workers:
                tasks.add(self.workers[agent_name])
            for task in tasks:
                task.cancel()
            if tasks:
                await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
            session = await self.sessions.current(agent_name)
            if session is not None:
                await finish_cleanup(self.agent_wrapper.release_session(session.session_id))
            agent = await complete_in_thread(self.agents.delete, agent_name, self.now())
            self.cron.pop(agent_name, None)
            self.changed.set()
            return {"agent_name": agent.agent_name, "deleted": True, "files_retained": True}

    async def append(self, agent_name, events):
        """Store reception time separately from observation time; duplicate inputs do not renew idle life."""
        async with self.management_lock:
            agent = await self.definition(agent_name)
            if agent.agent_type != "proactive":
                raise ValueError("external observations require an event agent")
            values = events if isinstance(events, list) else [events]
            values = [Event.model_validate(value) for value in values]
            if any(value.source == "cron" for value in values):
                raise ValueError("cron is a reserved framework source")
            return await self._append(agent, values)

    async def _append(self, agent, events: list[Event]):
        await self.prepare(agent)
        received = self.now().isoformat()
        values = [
            value.model_copy(update={"metadata": {**value.metadata, "_received_at": received}}) for value in events
        ]
        identifiers = await self.inbox.append(agent.agent_name, values)
        self.changed.set()
        return {"agent_name": agent.agent_name, "event_ids": identifiers}

    async def status(self, agent_name):
        """Inspect without acquiring a model-generation lock."""
        return await self._status(await self.definition(agent_name))

    async def _status(self, agent):
        agent_name = agent.agent_name
        await self.prepare(agent)
        snapshot = await self.inbox.snapshot(agent_name)
        receipt = snapshot["receipt"]
        execution = self.executions[agent_name]
        task_path = self.agents.task_path(agent)
        next_trigger = None
        if agent.enabled and agent.agent_type == "cron":
            schedule = self.cron.get(agent_name)
            signature = f"{agent.cron}|{self.app_context.app_config.timezone}"
            next_trigger = (
                schedule[1] if schedule and schedule[0] == signature else self._next_cron_time(agent, self.now())
            )
        return {
            "agent_name": agent_name,
            "config": agent.model_dump(mode="json", exclude={"created_at", "deleted_at"}),
            "created_at": agent.created_at.isoformat(),
            "task_path": task_path.relative_to(self.workspace_path).as_posix(),
            "running": agent_name in self.workers,
            "pending_events": snapshot["pending_count"],
            "in_flight_events": snapshot["in_flight_count"],
            "last_started_at": (execution.started_at.isoformat() if execution.started_at else None),
            "last_completed_at": receipt.consume_ts.isoformat() if receipt else None,
            "next_trigger_at": next_trigger.isoformat() if next_trigger is not None else None,
            "error": execution.error,
        }

    async def list_agents(self):
        """List task identities independently of chat and native session counts."""
        agents = await asyncio.to_thread(self.agents.load)
        items = [await self._status(agent) for agent in agents.values()]
        return {"items": items, "total": len(items)}

    @asynccontextmanager
    async def chat_turn(self):
        """Serialize Chat and explicit session rotation while owning shutdown cancellation."""
        if self.closed:
            raise RuntimeError("Agent runtime is closed")
        task = asyncio.current_task()
        self.notifications.foreground()
        self.chat_tasks.add(task)
        try:
            async with self.chat_lock:
                try:
                    yield
                finally:
                    self.notifications.activity()
        finally:
            self.chat_tasks.discard(task)

    async def list_views(self):
        """Inspect all visible identities, including Remy, without creating sessions."""
        from ._internal.views import AgentViews

        return await AgentViews(self).list()

    async def read_agent(self, agent_name):
        """Read the current formal session without activating a temporary query fork."""
        from ._internal.views import AgentViews

        return await AgentViews(self).read(agent_name)

    async def watch_agent(self, agent_name):
        """Stream reconstructible snapshots and changes; reconnect starts with a snapshot."""
        from ._internal.views import AgentViews

        async with aclosing(AgentViews(self).events(agent_name)) as events:
            async for event in events:
                yield event

    async def list_inputs(self, agent_name, **kwargs):
        """Page the same event Inbox for Chat and task Agents."""
        if agent_name != "Remy":
            await self.definition(agent_name)
        page = await self.inbox.read(agent_name, **kwargs)
        page["items"] = [{"event": item.pop("item"), **item} for item in page["items"]]
        if agent_name == "Remy":
            await self.outbox.refresh()
            refs = {ref.id: ref for ref in self.outbox.committed}
            for item in page["items"]:
                if ref := refs.get(item["event"]["id"]):
                    item["reminder"] = self.outbox.message(ref)
        return page

    async def list_chat_sessions(self, offset=0, limit=10):
        """Listing is side-effect-free, including before the first Chat turn."""
        return await self.sessions.list("Remy", offset, limit)

    @asynccontextmanager
    async def _agent_task(self, agent_name):
        """Own Agent calls before they wait for locks so shutdown and deletion can cancel them."""
        if self.closed:
            raise RuntimeError("Agent runtime is closed")
        task = asyncio.current_task()
        self.agent_tasks[task] = agent_name
        try:
            yield
        finally:
            self.agent_tasks.pop(task, None)

    @asynccontextmanager
    async def query_session(self, agent_name, options):
        """Own a temporary fork of the latest task session, including during execution."""
        async with self._agent_task(agent_name):
            async with self.management_lock:
                await self.definition(agent_name)
                source = await self.sessions.current(agent_name)
                if source is None:
                    raise ValueError("Task has no queryable session yet")
            session_id = None
            try:

                async def fork():
                    nonlocal session_id
                    session_id = await self.agent_wrapper.create_session(
                        resume=source.session_id,
                        fork_session=True,
                        **options,
                    )
                    await self.sessions.record(agent_name, session_id, self.now(), temporary=True)

                await finish_cleanup(fork())
                yield session_id
            finally:
                if session_id is not None:
                    await finish_cleanup(self.agent_wrapper.release_session(session_id))

    async def run(self, agent_name, execute):
        """Commit successful execution before handing asynchronous delivery to the outbox."""
        async with (
            self._agent_task(agent_name),
            self.round_locks.setdefault(agent_name, asyncio.Lock()),
        ):
            async with self.management_lock:
                agent = await self.definition(agent_name)
                if not agent.enabled:
                    return {"agent_name": agent_name, "skipped": "disabled"}
                await self.prepare(agent)
                execution = self.executions[agent_name]
                if execution.retry_items:
                    receipt = (await self.inbox.snapshot(agent_name))["receipt"]
                    if receipt and set(execution.retry_items).issubset(receipt.item_ids):
                        execution.retry_items = None
                batch = await self.inbox.claim(
                    agent_name,
                    limit=1 if agent.agent_type == "cron" else agent.max_batch_events,
                    item_ids=execution.retry_items,
                )
                if batch is None:
                    return {"agent_name": agent_name, "processed": 0}
                execution.retry_items = [item["id"] for item in batch["items"]]
                owns_worker = agent_name not in self.workers
                if owns_worker:
                    self.workers[agent_name] = asyncio.current_task()
            try:
                self.executions[agent_name].started_at = self.now()
                self.activities[agent_name] = AgentActivity(
                    session_id="",
                    turn_id=batch["token"],
                    source="task",
                    task_id=agent_name,
                    started_at=self.executions[agent_name].started_at,
                    phase="working",
                )
                result, session_id = await execute(agent, batch["items"])
                self.activities[agent_name] = self.activities[agent_name].model_copy(update={"session_id": session_id})
                run_id = sha256(json.dumps([agent_name, batch["items"]], sort_keys=True).encode()).hexdigest()[:32]

                async def commit():
                    previous = (await self.inbox.snapshot(agent_name))["receipt"]
                    received = [event["metadata"]["_received_at"] for event in batch["items"]]
                    if previous is not None:
                        received.append(previous.metadata["last_received_at"])
                    last_received = max(datetime.fromisoformat(value) for value in received)
                    refs = await self.outbox.save(agent, run_id, result, self.now())
                    await self.inbox.ack(
                        agent_name,
                        batch["token"],
                        metadata={
                            "reminders": [ref.model_dump() for ref in refs],
                            "session_id": session_id,
                            "last_received_at": last_received.isoformat(),
                        },
                        retain=bool(refs),
                    )
                    execution.retry_items = None

                await finish_cleanup(commit())
                self.logger.info(
                    f"Proactive agent={agent_name} session={session_id} result=committed "
                    f"processed={len(batch['items'])} notifications={len(result.notifications)}",
                )
                self.activities[agent_name] = self.activities[agent_name].model_copy(update={"phase": "completed"})
                self.changed.set()
                return {
                    "agent_name": agent_name,
                    "processed": len(batch["items"]),
                    "notifications": result.model_dump()["notifications"],
                }
            except BaseException as exc:
                phase = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
                self.activities[agent_name] = self.activities[agent_name].model_copy(update={"phase": phase})
                raise
            finally:
                try:
                    await finish_cleanup(self.inbox.nack(agent_name, batch["token"]))
                except ValueError:
                    pass
                if owns_worker:
                    self.workers.pop(agent_name, None)

    def _next_cron_time(self, agent, now):
        """Compute a calendar trigger without mutating the scheduler or enqueuing work."""
        from croniter import croniter

        anchor = now if self.app_context.app_config.timezone else now.replace(tzinfo=None)
        next_fire = croniter(agent.cron, anchor).get_next(datetime)
        if next_fire.tzinfo is None:
            next_fire = next_fire.astimezone()
        return next_fire

    async def _schedule_cron(self, agent):
        now = self.now()
        signature = f"{agent.cron}|{self.app_context.app_config.timezone}"
        next_fire = self._next_cron_time(agent, now)
        if agent.agent_name not in self.cron or self.cron[agent.agent_name][0] != signature:
            self.cron[agent.agent_name] = (signature, next_fire)
        scheduled = self.cron[agent.agent_name][1]
        if scheduled.timestamp() > now.timestamp():
            return
        await self._append(
            agent,
            [
                Event(
                    id=f"cron-{agent.agent_name}-{int(scheduled.timestamp())}",
                    event_ts=scheduled,
                    source="cron",
                    description=f"Scheduled trigger: {scheduled.isoformat()}",
                    metadata={"scheduled_at": scheduled.isoformat()},
                ),
            ],
        )
        self.cron[agent.agent_name] = (signature, next_fire)

    async def tick(self, run_job):
        """Check both consumers even when one scheduler fails."""
        results = await asyncio.gather(
            self._tick_tasks(run_job),
            self.notifications.tick(run_job),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _tick_tasks(self, run_job):
        """Schedule independent serial agents; idle cleanup runs under the same management lock."""
        async with self.management_lock:
            agents = await asyncio.to_thread(self.agents.load)
            for agent in agents.values():
                await self.prepare(agent)
                if agent.enabled and agent.agent_type == "cron":
                    await self._schedule_cron(agent)
                execution = self.executions[agent.agent_name]
                if not agent.enabled or agent.agent_name in self.workers:
                    continue
                if execution.retry_at is not None and execution.retry_at > self.now():
                    continue
                snapshot = await self.inbox.snapshot(agent.agent_name)
                receipt = snapshot["receipt"]
                if agent.agent_type == "proactive" and receipt is not None:
                    if (self.now() - receipt.consume_ts).total_seconds() < agent.min_trigger_interval_seconds:
                        continue
                if snapshot["pending_count"]:
                    self.workers[agent.agent_name] = asyncio.create_task(
                        self._work(agent.agent_name, run_job),
                        name=f"agent:{agent.agent_name}",
                    )
            for agent in (await asyncio.to_thread(self.agents.load, include_deleted=True)).values():
                if agent.temporary:
                    await self._collect(agent)

    async def _collect(self, agent):
        if agent.agent_name in self.workers or agent.agent_name in self.agent_tasks.values():
            return
        snapshot = await self.inbox.snapshot(agent.agent_name)
        receipt = snapshot["receipt"]
        received = datetime.fromisoformat(receipt.metadata["last_received_at"]) if receipt else agent.created_at
        if self.now() - received < timedelta(days=7):
            return
        if snapshot["pending_count"]:
            return
        # A worker can finish before the watcher sees its newly committed outbox.
        await self.outbox.sync()
        async with self.outbox.sync_lock, self.notifications.lock:
            refs = self.outbox.references(agent.agent_name)
            retained = set(self.notifications.pending.values()) | set(self.notifications.read_refs.values())
            if refs & retained or any(
                self.outbox.records[ref].delivered_at is None for ref in refs & self.outbox.committed
            ):
                return
            try:
                await self.inbox.forget(agent.agent_name)
            except ValueError:
                return

            async def remove():
                session = await self.sessions.current(agent.agent_name)
                if session is not None:
                    await self.agent_wrapper.release_session(session.session_id)
                await complete_in_thread(shutil.rmtree, self.agents.directory_for(agent))

            await finish_cleanup(remove())
        self.executions.pop(agent.agent_name, None)
        self.activities.pop(agent.agent_name, None)
        self.round_locks.pop(agent.agent_name, None)
        self.sessions.locks.pop(agent.agent_name, None)
        self.cron.pop(agent.agent_name, None)

    async def _work(self, agent_name, run_job):
        execution = self.executions[agent_name]
        try:
            response = await run_job("run_proactive_agent", agent_name=agent_name)
            if not response.success:
                raise RuntimeError(str(response.answer))
            execution.error = execution.retry_at = None
        except Exception as exc:
            execution.error = str(exc)[:1000]
            execution.retry_at = self.now() + timedelta(seconds=self.retry_seconds)
            self.logger.exception(f"Proactive agent {agent_name} failed: {exc}")
        finally:
            self.workers.pop(agent_name, None)
            self.changed.set()

    async def wait(self, stop, seconds=1):
        """Wake for external edits or new inputs while continuing timer checks."""
        tasks = [
            asyncio.create_task(stop.wait()),
            asyncio.create_task(self.changed.wait()),
        ]
        try:
            await asyncio.wait(tasks, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.changed.clear()
