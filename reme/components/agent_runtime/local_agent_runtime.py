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
from ._internal.inbox import AgentInbox
from ._internal.outbox import AgentOutbox
from ._internal.registry import AgentRegistry
from ._internal.sessions import AgentSessions
from ..agent_wrapper import BaseAgentWrapper
from ..event_stream import BaseEventStream
from ..component_registry import R
from ...schema.agent_runtime import ChatAgentConfig, CreateChatAgentConfig, InboxConfig
from ...schema.application_config import ApplicationConfig
from ...schema.event import Event
from ...schema.proactive_agent import CreateProactiveAgentConfig
from ...utils.async_utils import complete_in_thread, finish_cleanup

_MEMORY_PREVIEW_CHARS = 2000


@dataclass
class _Execution:
    started_at: datetime | None = None
    error: str | None = None
    retry_at: datetime | None = None


@R.register("local")
class LocalAgentRuntime(BaseAgentRuntime):
    """Own task lifetimes and rebuild execution facts from their source logs."""

    def __init__(
        self,
        agent_wrapper="default",
        event_stream="default",
        inbox=None,
        retry_seconds=5,
        agents=None,
        main=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if type(retry_seconds) not in (int, float) or not math.isfinite(retry_seconds) or retry_seconds < 0:
            raise ValueError("retry_seconds must be a finite nonnegative number")
        if agents is not None and not isinstance(agents, list):
            raise ValueError("agents must be a list of initial task configurations")
        self.initial_agents = [CreateProactiveAgentConfig.model_validate(value) for value in agents or []]
        if len({agent.name.casefold() for agent in self.initial_agents}) != len(self.initial_agents):
            raise ValueError("initial Agent names must be unique, including case")
        self.initial_main = CreateChatAgentConfig.model_validate(main if main is not None else {})
        self.agent_wrapper = self.bind(agent_wrapper, BaseAgentWrapper, optional=False)
        self.event_stream = self.bind(event_stream, BaseEventStream, optional=False)
        self.files = AgentFiles(self.workspace_path)
        directory = (
            self.app_context.app_config.reme_agent_dir if self.app_context else ApplicationConfig().reme_agent_dir
        )
        self.agents = AgentRegistry(self.files, directory)
        self.sessions = AgentSessions(self.agents)
        self.outbox = AgentOutbox(self)
        self.inbox = AgentInbox(self, InboxConfig.model_validate(inbox if inbox is not None else {}))
        self.retry_seconds = retry_seconds
        self.chat_lock = asyncio.Lock()
        self.chat_tasks = set()
        self.workers = {}
        self.queries = {}
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
        main = self.agents.path("main") / "config.json"
        config = self.files.read_json(main, None)
        config = (
            ChatAgentConfig.model_validate(config) if config is not None else ChatAgentConfig(created_at=self.now())
        )
        await complete_in_thread(
            self.agents.initialize_memory,
            main.parent,
            self.initial_main.description,
            language=self.app_context.app_config.language,
        )
        await complete_in_thread(self.files.write_json, main, config.model_dump(mode="json"))
        existing = {name.casefold() for name in await asyncio.to_thread(self.agents.load, include_deleted=True)}
        for initial in self.initial_agents:
            if initial.name.casefold() not in existing:
                await self.create(initial)
        for agent in (await asyncio.to_thread(self.agents.load)).values():
            await self.prepare(agent)
        await self.outbox.sync()
        self.inbox.last_activity_at = self.now()
        self.inbox.retry_at = None
        self.closed = False
        self.changed.set()
        self.watch_stop.clear()
        self.watcher = asyncio.create_task(self._watch(), name="agents:watch")

    async def _close(self):
        self.closed = True
        self.watch_stop.set()
        self.changed.set()
        await self.inbox.close()
        tasks = set(self.workers.values()) | set(self.queries) | self.chat_tasks
        if self.watcher is not None:
            tasks.add(self.watcher)
        for task in tasks:
            task.cancel()
        if tasks:
            await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
        self.workers.clear()
        self.queries.clear()
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
        """Mount one per-agent stream without creating independent component lifetimes."""
        directory = (self.agents.directory_for(agent) / "events").relative_to(self.workspace_path).as_posix()
        await self.event_stream.mount(agent.agent_id, directory)
        self.executions.setdefault(agent.agent_id, _Execution())

    async def definition(self, agent_id):
        """Read current user-owned configuration rather than a stale runtime copy."""
        return await asyncio.to_thread(self.agents.get, agent_id)

    async def create(self, config):
        """Persist a task and its event directory before returning success."""
        async with self.management_lock:
            agent = await complete_in_thread(
                self.agents.create,
                config,
                self.now(),
                language=self.app_context.app_config.language,
            )
            await self.prepare(agent)
            self.changed.set()
            return await self._status(agent, include_memory=True)

    async def update(self, agent_id, config):
        """Apply execution settings to future rounds and invalidate the in-memory timer."""
        async with self.management_lock:
            agent = await complete_in_thread(self.agents.update, agent_id, config)
            self.cron.pop(agent_id, None)
            await self.prepare(agent)
            self.changed.set()
            return await self._status(agent)

    async def delete(self, agent_id):
        """Stop owned work while retaining a restorable identity and all source files."""
        async with self.management_lock:
            tasks = {task for task, owner in self.queries.items() if owner == agent_id}
            if agent_id in self.workers:
                tasks.add(self.workers[agent_id])
            for task in tasks:
                task.cancel()
            if tasks:
                await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
            session = await self.sessions.current(agent_id)
            if session is not None:
                await finish_cleanup(self.agent_wrapper.release_session(session.session_id))
            agent = await complete_in_thread(self.agents.delete, agent_id, self.now())
            self.cron.pop(agent_id, None)
            self.changed.set()
            return {"agent_id": agent.agent_id, "deleted": True, "files_retained": True}

    async def append(self, agent_id, events):
        """Store reception time separately from observation time; duplicate inputs do not renew idle life."""
        async with self.management_lock:
            agent = await self.definition(agent_id)
            return await self._append(agent, events)

    async def _append(self, agent, events):
        await self.prepare(agent)
        values = events if isinstance(events, list) else [events]
        received = self.now().isoformat()
        values = [Event.model_validate(value) for value in values]
        values = [
            value.model_copy(update={"metadata": {**value.metadata, "_received_at": received}}) for value in values
        ]
        identifiers = await self.event_stream.append(agent.agent_id, values)
        self.changed.set()
        return {"agent_id": agent.agent_id, "event_ids": identifiers}

    async def snapshot(self, agent_id):
        """Recover the latest input and successful round from retained event logs."""
        return await self.event_stream.snapshot(agent_id)

    async def status(self, agent_id):
        """Inspect without acquiring a model-generation lock."""
        return await self._status(await self.definition(agent_id), include_memory=True)

    async def _status(self, agent, *, include_memory=False):
        agent_id = agent.agent_id
        await self.prepare(agent)
        snapshot = await self.snapshot(agent_id)
        receipt = snapshot["consumption"]
        execution = self.executions[agent_id]
        memory = self.agents.directory_for(agent) / "MEMORY.md"
        result = {
            "agent_id": agent_id,
            "config": agent.model_dump(mode="json", exclude={"created_at", "deleted_at"}),
            "created_at": agent.created_at.isoformat(),
            "memory_path": memory.relative_to(self.workspace_path).as_posix(),
            "running": agent_id in self.workers,
            "pending_events": snapshot["pending_count"],
            "in_flight_events": snapshot["in_flight_count"],
            "last_started_at": execution.started_at.isoformat() if execution.started_at else None,
            "last_completed_at": receipt.consume_ts.isoformat() if receipt else None,
            "next_trigger_at": self.cron[agent_id][1].isoformat() if agent_id in self.cron else None,
            "error": execution.error,
        }
        if include_memory:
            content = await asyncio.to_thread(self.files.read, memory)
            result.update(
                memory_content=content[:_MEMORY_PREVIEW_CHARS],
                memory_truncated=len(content) > _MEMORY_PREVIEW_CHARS,
            )
        return result

    async def list_agents(self):
        """List task identities independently of main and native session counts."""
        agents = await asyncio.to_thread(self.agents.load)
        items = [await self._status(agent) for agent in agents.values()]
        return {"items": items, "total": len(items)}

    @asynccontextmanager
    async def main_turn(self):
        """Serialize Chat and explicit session rotation while owning shutdown cancellation."""
        if self.closed:
            raise RuntimeError("Agent runtime is closed")
        task = asyncio.current_task()
        self.chat_tasks.add(task)
        try:
            async with self.chat_lock:
                try:
                    yield
                finally:
                    self.inbox.activity()
        finally:
            self.chat_tasks.discard(task)

    async def list_chat_sessions(self, offset=0, limit=10):
        """Listing is side-effect-free, including before the first Chat turn."""
        return await self.sessions.list("main", offset, limit)

    @asynccontextmanager
    async def query_turn(self, agent_id):
        """Serialize query registration with deletion and temporary Agent collection."""
        task = asyncio.current_task()
        try:
            async with self.management_lock:
                if self.closed:
                    raise RuntimeError("Agent runtime is closed")
                self.queries[task] = agent_id
                agent = await self.definition(agent_id)
                await self.prepare(agent)
                receipt = (await self.snapshot(agent_id))["consumption"]
                session_id = receipt.metadata.get("session_id") if receipt else None
                if session_id is None:
                    raise ValueError("proactive agent has no completed session to query")
            yield session_id
        finally:
            self.queries.pop(task, None)

    async def run(self, agent_id, execute):
        """Commit successful execution before handing asynchronous delivery to the outbox."""
        async with self.round_locks.setdefault(agent_id, asyncio.Lock()):
            async with self.management_lock:
                agent = await self.definition(agent_id)
                if not agent.enabled:
                    return {"agent_id": agent_id, "skipped": "disabled"}
                await self.prepare(agent)
                name = agent_id
                batch = await self.event_stream.claim(name, limit=1 if agent.trigger_type == "cron" else None)
                if batch is None:
                    return {"agent_id": agent_id, "processed": 0}
                owns_worker = agent_id not in self.workers
                self.workers[agent_id] = asyncio.current_task()
            try:
                self.executions[agent_id].started_at = self.now()
                result, session_id = await execute(agent, batch["events"])
                run_id = sha256(json.dumps([agent_id, batch["events"]], sort_keys=True).encode()).hexdigest()[:32]

                async def commit():
                    refs = await self.outbox.save(agent, run_id, result, self.now())
                    await self.event_stream.ack(
                        name,
                        batch["token"],
                        metadata={"reminders": [ref.model_dump() for ref in refs], "session_id": session_id},
                        retain=bool(refs),
                    )

                await finish_cleanup(commit())
                self.changed.set()
                return {
                    "agent_id": agent_id,
                    "processed": len(batch["events"]),
                    "notifications": result.model_dump()["notifications"],
                }
            finally:
                try:
                    await finish_cleanup(self.event_stream.nack(name, batch["token"]))
                except ValueError:
                    pass
                if owns_worker:
                    self.workers.pop(agent_id, None)

    async def _schedule_cron(self, agent):
        from croniter import croniter

        now = self.now()
        signature = f"{agent.cron}|{self.app_context.app_config.timezone}"
        anchor = now if self.app_context.app_config.timezone else now.replace(tzinfo=None)
        next_fire = croniter(agent.cron, anchor).get_next(datetime)
        if next_fire.tzinfo is None:
            next_fire = next_fire.astimezone()
        if agent.agent_id not in self.cron or self.cron[agent.agent_id][0] != signature:
            self.cron[agent.agent_id] = (signature, next_fire)
        scheduled = self.cron[agent.agent_id][1]
        if scheduled.timestamp() > now.timestamp():
            return
        await self._append(
            agent,
            Event(
                id=f"cron-{agent.agent_id}-{int(scheduled.timestamp())}",
                event_ts=scheduled,
                source="cron",
                description=f"Scheduled trigger: {scheduled.isoformat()}",
                metadata={"scheduled_at": scheduled.isoformat()},
            ),
        )
        self.cron[agent.agent_id] = (signature, next_fire)

    async def tick(self, run_job, job_name):
        """Schedule independent serial agents; idle cleanup runs under the same management lock."""
        async with self.management_lock:
            agents = await asyncio.to_thread(self.agents.load)
            for agent in agents.values():
                await self.prepare(agent)
                if agent.enabled and agent.trigger_type == "cron":
                    await self._schedule_cron(agent)
                execution = self.executions[agent.agent_id]
                if not agent.enabled or agent.agent_id in self.workers:
                    continue
                if execution.retry_at is not None and execution.retry_at > self.now():
                    continue
                snapshot = await self.snapshot(agent.agent_id)
                receipt = snapshot["consumption"]
                if agent.trigger_type == "event" and receipt is not None:
                    if (self.now() - receipt.consume_ts).total_seconds() < agent.min_trigger_interval_seconds:
                        continue
                if snapshot["pending_count"]:
                    self.workers[agent.agent_id] = asyncio.create_task(
                        self._work(agent.agent_id, run_job, job_name),
                        name=f"agent:{agent.agent_id}",
                    )
            for agent in (await asyncio.to_thread(self.agents.load, include_deleted=True)).values():
                if agent.temporary:
                    await self._collect(agent)

    async def _collect(self, agent):
        if agent.agent_id in self.workers or agent.agent_id in self.queries.values():
            return
        snapshot = await self.snapshot(agent.agent_id)
        event = snapshot["event"]
        received = datetime.fromisoformat(event["metadata"]["_received_at"]) if event else agent.created_at
        if self.now() - received < timedelta(days=7):
            return
        if snapshot["pending_count"]:
            return
        # A worker can finish before the watcher sees its newly committed outbox.
        await self.outbox.sync()
        async with self.outbox.sync_lock, self.inbox.lock:
            refs = self.outbox.references(agent.agent_id)
            retained = set(self.inbox.pending.values()) | set(self.inbox.read_refs.values())
            if refs & retained or any(
                self.outbox.records[ref].delivered_at is None for ref in refs & self.outbox.committed
            ):
                return

            async def remove():
                session = await self.sessions.current(agent.agent_id)
                if session is not None:
                    await self.agent_wrapper.release_session(session.session_id)
                await complete_in_thread(shutil.rmtree, self.agents.directory_for(agent))
                await self.event_stream.forget(agent.agent_id)

            await finish_cleanup(remove())
        self.executions.pop(agent.agent_id, None)
        self.round_locks.pop(agent.agent_id, None)
        self.sessions.locks.pop(agent.agent_id, None)
        self.cron.pop(agent.agent_id, None)

    async def _work(self, agent_id, run_job, job_name):
        execution = self.executions[agent_id]
        try:
            response = await run_job(job_name, agent_id=agent_id)
            if not response.success:
                raise RuntimeError(str(response.answer))
            execution.error = execution.retry_at = None
        except Exception as exc:
            execution.error = str(exc)[:1000]
            execution.retry_at = self.now() + timedelta(seconds=self.retry_seconds)
            self.logger.exception(f"Proactive agent {agent_id} failed: {exc}")
        finally:
            self.workers.pop(agent_id, None)
            self.changed.set()

    async def wait(self, stop, seconds=1):
        """Wake for external edits or new inputs while continuing timer checks."""
        tasks = [asyncio.create_task(stop.wait()), asyncio.create_task(self.changed.wait())]
        try:
            await asyncio.wait(tasks, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.changed.clear()
