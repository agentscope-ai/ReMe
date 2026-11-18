"""Reconstructible UI views over runtime ownership and native session history."""

import asyncio
from time import monotonic

from ....schema.agent_view import AgentSessionView, AgentView
from ....schema.agent_session import SessionSnapshot, SessionStatus


class AgentViews:
    """Subscriptions own no generation tasks and never create or select sessions."""

    def __init__(self, runtime):
        self.runtime = runtime

    async def view(self, agent_name):
        """Read identity, native activity and original file references."""
        runtime = self.runtime
        if agent_name == "Remy":
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
            directory = runtime.agents.path("Remy")
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
            if agent_name == "Remy" or path.startswith(prefix) or path.startswith(str(directory) + "/")
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
            running=running or status == "active",
            status=status,
            session_id=session.session_id if session else None,
            directory=directory.relative_to(runtime.workspace_path).as_posix(),
            pending_events=facts.get("pending_events", 0),
            last_started_at=facts.get("last_started_at"),
            last_completed_at=facts.get("last_completed_at"),
            next_trigger_at=facts.get("next_trigger_at"),
            error=error,
            **document,
        )

    async def list(self):
        """Include the fixed chat identity before named tasks."""
        agents = await asyncio.to_thread(self.runtime.agents.load)
        items = [await self.view(agent_name) for agent_name in ["Remy", *agents]]
        return {
            "items": [item.model_dump(mode="json") for item in items],
            "total": len(items),
        }

    async def read(self, agent_name, *, include_live=True):
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
        value = AgentSessionView(agent=agent, session_id=agent.session_id, snapshot=snapshot).model_dump(mode="json")
        if value["snapshot"]:
            for message in value["snapshot"]["messages"]:
                message["notification_eligible"] = (
                    agent_name == "Remy"
                    and message["source"] == "proactive"
                    and message["role"] == "assistant"
                    and message["completed"]
                    and bool(message["content"].strip())
                )
        return self.runtime.live_turns.overlay(agent_name, value) if include_live else value

    async def events(self, agent_name, interval=1, heartbeat=15):
        """Diff bounded polling views; every new connection receives an authoritative snapshot.

        Polling also observes external file changes and works with every session-capable wrapper.
        Cancellation belongs to the request; observing never acquires a generation lock.
        """
        wake = asyncio.Event()
        self.runtime.live_turns.listeners.add(wake)
        try:
            previous: dict = {}
            last_emit = monotonic()
            next_poll = 0
            polled_epoch = -1
            value, agents = {}, []
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
                    value = await self.read(agent_name, include_live=False)
                    agents = (await self.list())["items"]
                    if epoch != self.runtime.live_turns.epoch:
                        continue
                    polled_epoch = epoch
                    next_poll = monotonic() + interval
                current = self.runtime.live_turns.overlay(agent_name, value)
                snapshot = current["snapshot"]
                activity = self.runtime.activities.get(agent_name)
                activities = (
                    {"chat": activity.model_dump(mode="json")}
                    if activity and activity.source == "chat" and activity.session_id == value["session_id"]
                    else {}
                )
                event = {
                    "type": "snapshot",
                    "agent_name": agent_name,
                    "session_id": value["session_id"],
                    "status": value["agent"]["status"],
                    "activities": activities,
                    "task_activities": {
                        name: activity.model_dump(mode="json")
                        for name, activity in self.runtime.activities.items()
                        if activity.source == "task" and name in self.runtime.workers
                    },
                    "agents": agents,
                    "snapshot": snapshot,
                }
                if event != previous:
                    # A task turn replaces its predecessor atomically, including its live partial output.
                    if not previous or agent_name != "Remy":
                        yield event
                    elif previous["session_id"] != event["session_id"]:
                        yield {**event, "type": "session_changed", "snapshot": None}
                        yield {**event, "type": "status_changed", "snapshot": None}
                        for message in (snapshot or {}).get("messages", []):
                            yield {
                                **event,
                                "type": "message_upsert",
                                "snapshot": None,
                                "message": message,
                            }
                    else:
                        old = {message["id"]: message for message in (previous["snapshot"] or {}).get("messages", [])}
                        messages = (snapshot or {}).get("messages", [])
                        ids = [message["id"] for message in messages]
                        if list(old) != ids[: len(old)]:
                            # Compaction/removal/reordering is one atomic replacement.
                            yield event
                        else:
                            if any(
                                event[key] != previous[key]
                                for key in (
                                    "status",
                                    "activities",
                                    "task_activities",
                                    "agents",
                                )
                            ):
                                yield {**event, "type": "status_changed", "snapshot": None}
                            for message in messages:
                                if message != old.get(message["id"]):
                                    yield {
                                        **event,
                                        "type": "message_upsert",
                                        "snapshot": None,
                                        "message": message,
                                    }
                    previous, last_emit = event, monotonic()
                elif monotonic() - last_emit >= heartbeat:
                    yield {
                        "type": "heartbeat",
                        "agent_name": agent_name,
                        "activities": {},
                        "agents": agents,
                    }
                    last_emit = monotonic()
                try:
                    await asyncio.wait_for(wake.wait(), timeout=max(0, next_poll - monotonic()))
                    await asyncio.sleep(0.025)
                except TimeoutError:
                    pass
        finally:
            self.runtime.live_turns.listeners.discard(wake)
