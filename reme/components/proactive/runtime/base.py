"""Lifecycle, sessions and execution protection shared by Agent Runtimes."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from ...base import BaseComponent
from ...agent_wrapper import BaseAgentWrapper
from ._live_turn import LiveTurns
from ._sessions import RuntimeSessions
from ....enumeration import ComponentEnum
from ....schema.proactive_runtime import validate_agent_name
from ....utils.async_utils import finish_cleanup
from ....utils.files.workspace_files import WorkspaceFiles
from ....utils.timezone_utils import resolve_timezone


class BaseAgentRuntime(BaseComponent):
    """Own state and reliable operations; Jobs own model execution and scheduling."""

    component_type = ComponentEnum.AGENT_RUNTIME

    def __init__(self, agent_wrapper="default", run_job="", **kwargs):
        super().__init__(**kwargs)
        validate_agent_name(self.name)
        self.agent_wrapper = (
            self.bind(agent_wrapper, BaseAgentWrapper, optional=False)
            if isinstance(agent_wrapper, str)
            else agent_wrapper
        )
        self.run_job = run_job
        self.files = WorkspaceFiles(self.workspace_path)
        root = self.app_context.app_config.reme_agent_dir if self.app_context else "reme_agents"
        self.directory = self.files.path(f"{root}/{self.name}")
        self.sessions = RuntimeSessions(self)
        self.live_turns = LiveTurns()
        self.execution_lock = asyncio.Lock()
        self.tasks = set()
        self.worker = None
        self.activity = None
        self.error = None
        self.last_started_at = None
        self.last_completed_at = None
        self.closed = True
        self.inbox = None
        self.outbox = None

    def now(self):
        """Return application-local time using the configured timezone rules."""
        timezone = self.app_context.app_config.timezone if self.app_context else None
        return datetime.now(resolve_timezone(timezone))

    @property
    def current_tasks(self):
        """Include every owned active worker and queued invocation."""
        return self.tasks | ({self.worker} if self.worker is not None else set())

    @asynccontextmanager
    async def track(self):
        """Register this invocation so deletion and shutdown can join it."""
        self._check_open()
        task = asyncio.current_task()
        nested = task in self.tasks
        self.tasks.add(task)
        try:
            yield
        finally:
            if not nested:
                self.tasks.discard(task)

    @asynccontextmanager
    async def execution(self):
        """Hold the Runtime's serial execution lease for one invocation."""
        async with self.track(), self.execution_lock:
            self._check_open()
            yield

    def _check_open(self):
        if self.closed:
            raise RuntimeError("Agent Runtime is closed")

    async def start_children(self):
        """Start owned Inbox and Outbox resources with rollback on failure."""
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            await self.inbox.start()
            await self.outbox.start()
        except BaseException:
            await asyncio.gather(self.outbox.close(), self.inbox.close(), return_exceptions=True)
            raise
        self.closed = False

    async def stop_work(self):
        """Cancel and join active and waiting invocations without closing shared dependencies."""
        current = asyncio.current_task()
        tasks = self.current_tasks - {current}
        for task in tasks:
            task.cancel()
        if tasks:
            await finish_cleanup(asyncio.gather(*tasks, return_exceptions=True))
        self.worker = None

    async def _close(self):
        self.closed = True
        await self.stop_work()
        self.live_turns.clear()
        try:
            identifier = self.sessions.active_session_id
            if identifier is None:
                try:
                    session = await self.sessions.current()
                    identifier = session.session_id if session else None
                except (OSError, ValueError) as exc:
                    self.logger.error(f"Cannot read session reference during shutdown: {exc}")
            if identifier is not None:
                await self.agent_wrapper.release_session(identifier)
        finally:
            errors = []
            for child in (self.outbox, self.inbox):
                if child is not None:
                    try:
                        await child.close()
                    except Exception as exc:
                        errors.append(exc)
            if errors:
                raise ExceptionGroup("Could not close Runtime resources", errors)

    async def current_session(self):
        """Read the primary session reference without creating a session."""
        return await self.sessions.current()

    async def ensure_session(self, options, *, new=False):
        """Prepare a primary session using the role's execution binding."""
        return await self.sessions.ensure(options, new=new)

    @asynccontextmanager
    async def query_session(self, options):
        """Own a temporary read-only query branch without replacing primary history."""
        async with self.track():
            source = await self.current_session()
            if source is None:
                raise ValueError("Task has no queryable session yet")
            identifier = None
            try:

                async def create():
                    nonlocal identifier
                    identifier = await self.agent_wrapper.create_session(
                        resume=source.session_id,
                        fork_session=True,
                        **options,
                    )
                    await self.sessions.record(identifier, temporary=True)

                await create()
                yield identifier
            finally:
                if identifier is not None:
                    await finish_cleanup(self.agent_wrapper.release_session(identifier))

    async def snapshot(self):
        """Read queue counts and the latest durable receipt without claiming."""
        return await self.inbox.snapshot()
