"""Reconstructible UI views over runtime ownership and native session history."""

import asyncio
from time import monotonic

from ....constants import CHAT_AGENT_NAME
from ....schema.agent_view import (
    AgentSessionView,
    AgentView,
    AgentSnapshot,
    AgentUpdate,
    AgentHeartbeat,
    ClientActivity,
)
from ....utils.agent_transcript import project_messages
from ....schema.agent_session import SessionSnapshot, SessionStatus


class AgentViews:
    """Subscriptions own no generation tasks and never create or select sessions."""

    def __init__(self, runtime):
        self.runtime = runtime

    async def view(self, agent_name):
        """Read identity, native activity and original file references."""
        runtime = self.runtime
        if agent_name == CHAT_AGENT_NAME:
            inbox = runtime.notifications
            config, facts, agent_type = (
                {},
                {
                    "error": inbox.error,
                    "last_started_at": inbox.last_started_at.isoformat() if inbox.last_started_at else None,
                    "last_completed_at": inbox.last_completed_at.isoformat() if inbox.last_completed_at else None,
                    "next_trigger_at": inbox.retry_at.isoformat() if inbox.retry_at else None,
                },
                "chat",
            )
            directory = runtime.agents.path(CHAT_AGENT_NAME)
            document = {"memory_path": runtime.agents.chat_memory_path().relative_to(runtime.workspace_path).as_posix()}
            running = runtime.chat_lock.locked() or inbox.worker is not None
        else:
            agent = await runtime.definition(agent_name)
            facts = await runtime.status(agent_name)
            config, agent_type = facts["config"], agent.agent_type
            directory = runtime.agents.directory_for(agent)
            document = {"task_path": facts["task_path"]}
            running = facts["running"]
        session = await runtime.sessions.current(agent_name)
        status, error = "idle", facts.get("error")
        issues = runtime.outbox.issues
        prefix = directory.relative_to(runtime.workspace_path).as_posix() + "/"
        problems = [
            f"{path}: {message}"
            for path, message in issues.items()
            if agent_name == CHAT_AGENT_NAME or path.startswith(prefix) or path.startswith(str(directory) + "/")
        ]
        if problems:
            error = "; ".join(filter(None, [error, *problems]))[:2000]
        if session:
            try:
                if await runtime.agent_wrapper.session_exists(session.session_id):
                    status = (await runtime.agent_wrapper.get_session_status(session.session_id)).status
                else:
                    status = "not_loaded"
            except Exception as exc:  # A missing native thread must not hide the Agent's files.
                status, error = "error", str(exc)
        if running:
            status = "active"
        elif error:
            status = "error"
        return AgentView(
            agent_name=agent_name,
            agent_type=agent_type,
            enabled=config.get("enabled", True),
            cron=config.get("cron"),
            min_trigger_interval_seconds=config.get("min_trigger_interval_seconds", 0),
            notify_window=config.get("notify_window"),
            status=status,
            session_id=session.session_id if session else None,
            directory=directory.relative_to(runtime.workspace_path).as_posix(),
            pending_events=facts.get("pending_events", 0),
            in_flight_events=facts.get("in_flight_events", 0),
            last_started_at=facts.get("last_started_at"),
            last_completed_at=facts.get("last_completed_at"),
            next_trigger_at=facts.get("next_trigger_at"),
            error=error,
            **document,
        )

    async def list(self):
        """Include the fixed chat identity before named tasks."""
        agents = await asyncio.to_thread(self.runtime.agents.load)
        items = [await self.view(agent_name) for agent_name in [CHAT_AGENT_NAME, *agents]]
        return {
            "items": [item.model_dump(mode="json") for item in items],
            "total": len(items),
        }

    async def native(self, agent_name):
        """Keep Chat history; task workspaces expose only their latest native turn."""
        agent = await self.view(agent_name)
        snapshot = None
        if agent.session_id is not None:
            if await self.runtime.agent_wrapper.session_exists(agent.session_id):
                snapshot = await self.runtime.agent_wrapper.read_session(agent.session_id)
                if agent.agent_type != "chat":
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
                # Native empty threads may have no rollout until the first turn. Keep the
                # application reference visible; the next send will ensure a usable thread.
                snapshot = SessionSnapshot(
                    session_id=agent.session_id,
                    state=SessionStatus(session_id=agent.session_id, status="not_loaded"),
                )
        return {
            "agent": agent.model_dump(mode="json"),
            "session_id": agent.session_id,
            "snapshot": snapshot.model_dump(mode="json") if snapshot else None,
        }

    def project(self, agent_name, value):
        """Keep native reconciliation internal; publish only canonical display messages."""
        current = self.runtime.live_turns.overlay(agent_name, value)
        activities = {}
        for name, activity in self.runtime.activities.items():
            if name != agent_name and (agent_name != CHAT_AGENT_NAME or name not in self.runtime.workers):
                continue
            if name == agent_name and activity.session_id != current["session_id"]:
                continue
            activities[name] = ClientActivity(
                session_id=activity.session_id,
                turn_id=activity.turn_id,
                phase=activity.phase,
                tool=activity.tool,
                started_at=activity.started_at.isoformat() if activity.started_at else None,
            )
        return AgentSessionView(
            agent=current["agent"],
            messages=project_messages(
                current["session_id"] or "",
                (current["snapshot"] or {}).get("messages", []),
                notifications=agent_name == CHAT_AGENT_NAME,
            ),
            activities=activities,
        ).model_dump(mode="json")

    async def read(self, agent_name):
        """Read exactly the state used by the subscription's authoritative snapshot."""
        return self.project(agent_name, await self.native(agent_name))

    async def events(self, agent_name, interval=1, heartbeat=15):
        """Replace on reconnect/session change; otherwise batch complete-message upserts.

        Polling observes external edits. Live output wakes observers without rereading native files.
        Only the main subscription carries the global Agent roster and background activities.
        """
        wake = asyncio.Event()
        self.runtime.live_turns.listeners.add(wake)
        try:
            previous: dict = {}
            last_emit, next_poll, polled_epoch = monotonic(), 0, -1
            value: dict = {}
            agents = None
            while not self.runtime.closed:
                wake.clear()
                live = self.runtime.live_turns.turns.get(agent_name)
                epoch = self.runtime.live_turns.epoch
                if (
                    not value
                    or monotonic() >= next_poll
                    or epoch != polled_epoch
                    or (live and live.session_id != value["session_id"])
                ):
                    value = await self.native(agent_name)
                    agents = (await self.list())["items"] if agent_name == CHAT_AGENT_NAME else None
                    if epoch != self.runtime.live_turns.epoch:
                        continue
                    polled_epoch, next_poll = epoch, monotonic() + interval
                current = {**self.project(agent_name, value), "agents": agents}
                if current != previous:
                    old = {message["id"]: message for message in previous["messages"]} if previous else {}
                    messages = current["messages"]
                    ids = [message["id"] for message in messages]
                    replace = (
                        not previous
                        or previous["agent"]["session_id"] != current["agent"]["session_id"]
                        or list(old) != ids[: len(old)]
                    )
                    if replace:
                        yield AgentSnapshot.model_validate(current).model_dump(mode="json")
                    else:
                        changed = [message for message in messages if message != old.get(message["id"])]
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
            self.runtime.live_turns.listeners.discard(wake)
