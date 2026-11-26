"""Isolated real component assembly with an in-memory native Agent backend."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

from contextlib import asynccontextmanager

from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.job import BaseJob
from reme.components.proactive import ChatAgentRuntime, LocalAgentManager
from reme.config import resolve_app_config
from reme.core.context import ApplicationContext
from reme.schema import SessionMessage, SessionSnapshot, SessionStatus, StreamChunk
from reme.steps.proactive.task.initialize import initialize_task


class FakeWrapper(BaseAgentWrapper):
    """Keep native sessions, revisions and inference entirely inside a test workspace."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.created = []
        self.turns = []
        self.messages = {}
        self.released = []
        self.states = {}

    async def create_session(self, **kwargs):
        self.created.append(kwargs)
        identifier = f"session-{len(self.created)}"
        self.messages[identifier] = list(self.messages.get(kwargs.get("resume"), []))
        return identifier

    async def release_session(self, session_id):
        self.released.append(session_id)

    async def session_exists(self, session_id):
        return session_id in self.messages

    async def get_session_status(self, session_id):
        return SessionStatus(session_id=session_id, status=self.states.get(session_id, "idle"))

    async def read_session(self, session_id):
        return SessionSnapshot(
            session_id=session_id,
            state=await self.get_session_status(session_id),
            messages=self.messages.get(session_id, []),
            revision=str(len(self.messages.get(session_id, []))),
        )

    @asynccontextmanager
    async def session_guard(self, session_id, *, expected_revision):
        if (await self.read_session(session_id)).revision != expected_revision:
            raise RuntimeError("Session changed before committing decision")
        yield

    async def append_session(
        self,
        session_id,
        content,
        *,
        message_id,
        role="assistant",
        expected_revision=None,
        **_kwargs,
    ):
        existing = next((row for row in self.messages[session_id] if row.id == message_id), None)
        if existing:
            if existing.content != content:
                raise ValueError("Message id conflict")
            return existing
        if expected_revision is not None and (await self.read_session(session_id)).revision != expected_revision:
            raise RuntimeError("Session changed before native append")
        message = SessionMessage(id=message_id, role=role, content=content)
        self.messages[session_id].append(message)
        return message

    async def reply_stream(self, inputs, **kwargs):
        self.turns.append((inputs, kwargs))
        result = {
            "session_id": kwargs["resume"],
            "result": "Processed",
            "structured_output": None,
            "usage": None,
            "status": "completed",
            "error": None,
            "raw": {},
        }
        yield StreamChunk(chunk_type="content", chunk="Processed")
        yield StreamChunk(chunk_type="reply_end", chunk=result, status="completed")


@asynccontextmanager
async def running_system(tmp_path, *, language="en", initial_memory="", runtimes=None):
    config = resolve_app_config(config="proactive", log_config=False)
    context = ApplicationContext(
        workspace_dir=str(tmp_path),
        timezone="Asia/Shanghai",
        language=language,
        jobs=config["jobs"],
    )
    wrapper = FakeWrapper(app_context=context)
    chat = ChatAgentRuntime(name="Remy", agent_wrapper=wrapper, initial_memory=initial_memory, app_context=context)
    manager = LocalAgentManager(name="default", agent_wrapper=wrapper, notification_target=chat, app_context=context)
    context.components = {
        "agent_wrapper": {"default": wrapper},
        "agent_runtime": {"Remy": chat},
        "agent_manager": {"default": manager},
    }
    # Actual base Jobs exercise fresh Step construction and service-to-internal dispatch.
    context.jobs = {
        name: BaseJob(name=name, app_context=context, **value)
        for name, value in config["jobs"].items()
        if value["backend"] == "base"
    }
    async with wrapper, chat, manager:
        for job in context.jobs.values():
            await job.start()
        try:
            for value in runtimes or []:
                if value["agent_name"] not in manager.runtimes:
                    await manager.create(value)
                await initialize_task(
                    await manager.resolve(value["agent_name"]),
                    context,
                    value.get("task_description"),
                )
            yield manager, context, wrapper
        finally:
            for job in reversed(list(context.jobs.values())):
                await job.close()


async def queue_event(manager, name="monitor", identifier="event-1"):
    from reme.schema import Event

    await manager.append_events(
        name,
        [Event(id=identifier, event_ts=manager.now(), source="test", description="Finished")],
    )


async def emit_notification(manager, _context, *, name="monitor", suffix=0, priority="P1", message="Verified result"):
    """Commit a real source receipt and hand it off to Remy's notification queue."""
    from reme.schema import Event, AgentReminder
    from reme.utils.proactive_record import RoundRecord

    if name not in manager.runtimes:
        await manager.create({"agent_name": name})
    runtime = await manager.resolve(name)
    event = Event(id=f"event-{suffix}", event_ts=runtime.now(), source="test", description="Observed")
    await manager.append_events(name, [event])
    batch = await runtime.inbox.claim(item_ids=[event.id])
    path = runtime.daily_directory / runtime.now().date().isoformat() / f"{name}_{suffix:06}.md"
    record = RoundRecord(runtime.files)
    await record.prepare(path)
    runtime.files.write(path, f"---\nreminder_priority: {priority}\nreminder_message: {message}\n---\nEvidence\n")
    record.capture()
    reminder = AgentReminder(priority=priority, message=message, id=runtime.to_workspace_relative(path))
    await runtime.commit_round(batch, reminder, "source-session", record)
    from reme.schema import NotificationRef

    return NotificationRef.model_validate((await runtime.inbox.receipts())[-1].metadata["notification"])
