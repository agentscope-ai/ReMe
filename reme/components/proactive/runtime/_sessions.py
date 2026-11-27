"""One Runtime's durable native-session references."""

import asyncio
from collections import deque
from hashlib import sha256
import json

from ....schema.proactive_runtime import AgentSession
from ....utils.async_utils import complete_in_thread, finish_cleanup


class RuntimeSessions:
    """Select primary sessions while retaining temporary forks for audit."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = asyncio.Lock()
        self.active_session_id = None

    @property
    def path(self):
        """Locate this Runtime's native-session reference journal."""
        return self.runtime.files.checked(self.runtime.directory / "sessions.jsonl")

    def _records(self):
        for value in self.runtime.files.iter_jsonl(self.path, incomplete_ok=True):
            yield AgentSession.model_validate(value)

    async def current(self):
        """Select the newest nontemporary native-session reference."""

        def latest():
            current = None
            for record in self._records():
                if not record.temporary:
                    current = record
            return current

        current = await asyncio.to_thread(latest)
        self.active_session_id = current.session_id if current else None
        return current

    async def list(self, offset=0, limit=10):
        """Page persisted references without opening a native session."""
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Invalid session pagination")

        def read():
            """Return an isolated read-only projection."""
            records = deque(maxlen=offset + limit)
            total = 0
            for record in self._records():
                records.append(record)
                total += 1
            return {
                "items": [
                    record.model_dump(mode="json", exclude={"signature"})
                    for record in list(reversed(records))[offset : offset + limit]
                ],
                "total": total,
                "offset": offset,
                "limit": limit,
            }

        return await asyncio.to_thread(read)

    def _record(self, record):
        if not any(previous.session_id == record.session_id for previous in self._records()):
            self.runtime.files.append_jsonl(self.path, record.model_dump(mode="json"))

    async def record(self, session_id, *, temporary=False):
        """Persist an independent native-session reference exactly once."""
        async with self.lock:
            await complete_in_thread(
                self._record,
                AgentSession(session_id=session_id, created_at=self.runtime.now(), temporary=temporary),
            )

    async def ensure(self, options, *, new=False):
        """Reuse matching Chat history or create the next primary session."""
        wrapper = self.runtime.agent_wrapper
        binding = {key: value for key, value in options.items() if key != "_skills"}
        binding["skills"] = [
            {key: value for key, value in spec.items() if key != "name"} if spec.get("content") else spec
            for spec in binding.get("skills", [])
        ]
        signature = sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        async with self.lock:
            previous = await self.current()
            exists = not new and previous is not None and await wrapper.session_exists(previous.session_id)
            if exists and previous.signature == signature:
                return previous.session_id

            arguments = dict(options)
            if exists:
                arguments.update(resume=previous.session_id, fork_session=True)
            identifier = await wrapper.create_session(**arguments)
            try:
                await complete_in_thread(
                    self._record,
                    AgentSession(session_id=identifier, created_at=self.runtime.now(), signature=signature),
                )
            except BaseException:
                await finish_cleanup(wrapper.release_session(identifier))
                raise
            self.active_session_id = identifier
            if previous is not None:
                try:
                    await wrapper.release_session(previous.session_id)
                except Exception as exc:
                    self.runtime.logger.warning(f"Could not release previous session: {exc}")
            return identifier
