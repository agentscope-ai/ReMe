"""Agent-owned native session histories, independent of task and inbox identities."""

import asyncio
from collections import deque
from hashlib import sha256
import json

from ....schema.proactive_runtime import AgentSession, validate_agent_name
from ....utils.async_utils import complete_in_thread, finish_cleanup


class AgentSessions:
    """Persist every created session; only primary sessions determine current execution."""

    def __init__(self, agents):
        self.agents = agents
        self.files = agents.files
        self.locks = {}

    def path(self, agent_name):
        """Keep registry files separate from native transcripts and standard dialog notes."""
        return self.files.checked(self.agents.path(agent_name) / "sessions.jsonl")

    def lock(self, agent_name):
        """Coordinate selection with new-session publication and native delivery."""
        validate_agent_name(agent_name)
        return self.locks.setdefault(agent_name, asyncio.Lock())

    def _records(self, path):
        for value in self.files.iter_jsonl(path, incomplete_ok=True):
            yield AgentSession.model_validate(value)

    def _latest(self, agent_name, directory):
        path = self.path(agent_name) if directory is None else self.files.checked(directory / "sessions.jsonl")
        current = None
        for record in self._records(path):
            if not record.temporary:
                current = record
        return current

    async def current(self, agent_name, *, directory=None):
        """Read the last primary reference without creating or restoring a native thread."""
        return await asyncio.to_thread(self._latest, agent_name, directory)

    async def list(self, agent_name, offset=0, limit=10):
        """Return newest-first history; pagination never prunes or activates sessions."""
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("offset must be a nonnegative integer and limit a positive integer")

        def read():
            records = deque(maxlen=offset + limit)
            total = 0
            for record in self._records(self.path(agent_name)):
                records.append(record)
                total += 1
            items = list(reversed(records))[offset : offset + limit]
            return {
                "items": [record.model_dump(mode="json", exclude={"signature"}) for record in items],
                "total": total,
                "offset": offset,
                "limit": limit,
            }

        return await asyncio.to_thread(read)

    def _record(self, agent_name, record):
        path = self.path(agent_name)
        if not any(previous.session_id == record.session_id for previous in self._records(path)):
            self.files.append_jsonl(path, record.model_dump(mode="json"))

    async def record(self, agent_name, session_id, now, *, temporary=False):
        """Retain temporary forks for audit without making them the next execution source."""
        async with self.lock(agent_name):
            record = AgentSession(session_id=session_id, created_at=now, temporary=temporary)
            await complete_in_thread(self._record, agent_name, record)

    async def ensure(self, agent_name, wrapper, options, now, *, new=False):
        """Select or create a session with the supplied instructions and bindings."""
        signature = sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
        async with self.lock(agent_name):
            previous = await self.current(agent_name)
            exists = not new and previous is not None and await wrapper.session_exists(previous.session_id)
            if exists and previous.signature == signature:
                return previous.session_id

            async def create():
                arguments = dict(options)
                if not new and exists:
                    arguments.update(resume=previous.session_id, fork_session=True)
                session_id = await wrapper.create_session(**arguments)
                try:
                    record = AgentSession(session_id=session_id, created_at=now, signature=signature)
                    await complete_in_thread(self._record, agent_name, record)
                except BaseException:
                    await finish_cleanup(wrapper.release_session(session_id))
                    raise
                if previous is not None:
                    try:
                        await wrapper.release_session(previous.session_id)
                    except Exception as exc:
                        wrapper.logger.warning(f"Could not release previous session: {exc}")
                return session_id

            return await finish_cleanup(create())
