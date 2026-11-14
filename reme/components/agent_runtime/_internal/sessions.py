"""Agent-owned native session histories, independent of task and inbox identities."""

import asyncio
from collections import deque
from hashlib import sha256
import json

from ....schema.agent_runtime import AgentSession, validate_agent_id
from ....utils.async_utils import complete_in_thread, finish_cleanup


class AgentSessions:
    """Persist every created session; only primary sessions determine current execution."""

    def __init__(self, agents):
        self.agents = agents
        self.files = agents.files
        self.locks = {}

    def path(self, agent_id):
        """Keep registry files separate from native transcripts and standard dialog notes."""
        return self.files.checked(self.agents.path(agent_id) / "sessions.jsonl")

    def lock(self, agent_id):
        """Coordinate selection with new-session publication and native delivery."""
        validate_agent_id(agent_id)
        return self.locks.setdefault(agent_id, asyncio.Lock())

    def _records(self, agent_id):
        for value in self.files.iter_jsonl(self.path(agent_id), incomplete_ok=True):
            yield AgentSession.model_validate(value)

    def _latest(self, agent_id):
        current = None
        for record in self._records(agent_id):
            if not record.temporary:
                current = record
        return current

    async def current(self, agent_id):
        """Read the last primary reference without creating or restoring a native thread."""
        return await asyncio.to_thread(self._latest, agent_id)

    async def list(self, agent_id, offset=0, limit=10):
        """Return newest-first history; pagination never prunes or activates sessions."""
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError("offset must be a nonnegative integer and limit a positive integer")

        def read():
            records = deque(maxlen=offset + limit)
            total = 0
            for record in self._records(agent_id):
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

    def _record(self, agent_id, record):
        if not any(previous.session_id == record.session_id for previous in self._records(agent_id)):
            self.files.append_jsonl(self.path(agent_id), record.model_dump(mode="json"))

    async def record(self, agent_id, session_id, now, *, temporary=False):
        """Retain temporary forks for audit without making them the next execution source."""
        async with self.lock(agent_id):
            record = AgentSession(session_id=session_id, created_at=now, temporary=temporary)
            await complete_in_thread(self._record, agent_id, record)

    async def ensure(self, agent_id, wrapper, options, now, *, new=False):
        """Create, resume, or fork before atomically selecting the native session."""
        signature = sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
        async with self.lock(agent_id):
            previous = await self.current(agent_id)
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
                    await complete_in_thread(self._record, agent_id, record)
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
