"""Selected Main conversation and durable consumption cursors."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
from typing import TYPE_CHECKING

from ...schema.pa_session import PaSessionCursor
from ...utils.async_utils import complete_in_thread, finish_cleanup

if TYPE_CHECKING:
    from .runtime import PaRuntime

_SESSION_LIMIT = 100
_SESSION_FILE_BYTES = 128 * 1024


class PaSessions:
    """Serialize selection and consume only completed conversation rounds."""

    def __init__(self, runtime: PaRuntime):
        self.runtime = runtime
        self._sessions_lock = asyncio.Lock()
        self._conversation_lock = asyncio.Lock()

    def load(self) -> list[PaSessionCursor]:
        """Read the bounded session registry without replacing invalid state."""
        try:
            with self.runtime.session_ids_path.open("rb") as handle:
                content = handle.read(_SESSION_FILE_BYTES + 1)
        except FileNotFoundError:
            return []
        if len(content) > _SESSION_FILE_BYTES:
            raise ValueError("session_ids.json exceeds 128 KiB")
        data = json.loads(content)
        if not isinstance(data, list) or len(data) > _SESSION_LIMIT:
            raise ValueError("session_ids.json must contain at most 100 session records")
        sessions = [PaSessionCursor.model_validate(item) for item in data]
        if len({item.session_id for item in sessions}) != len(sessions):
            raise ValueError("session_ids.json contains duplicate sessions")
        return sessions

    async def remember_session(self, session_id: str) -> None:
        """Move a selected chat to the end without discarding its read cursor."""
        async with self._sessions_lock:
            sessions = await asyncio.to_thread(self.load)
            current = next((item for item in sessions if item.session_id == session_id), None)
            active_at = self.runtime.now().isoformat()
            if current is None:
                current = PaSessionCursor(session_id=session_id, last_active_at=active_at)
            current.last_active_at = active_at
            sessions = [item for item in sessions if item.session_id != session_id] + [current]
            await complete_in_thread(
                self.runtime.files.write_json,
                self.runtime.session_ids_path,
                [item.model_dump() for item in sessions[-_SESSION_LIMIT:]],
            )

    async def read_conversation(
        self,
        max_turns: int = 10,
        max_message_chars: int = 2000,
        max_chars: int = 20000,
        *,
        round_id: str | None = None,
    ) -> str:
        """Consume recent unread chat rounds, keeping each user input with its replies."""
        for name, value, maximum in (
            ("max_turns", max_turns, 100),
            ("max_message_chars", max_message_chars, 32000),
            ("max_chars", max_chars, 32000),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 1 to {maximum}")

        def timestamp(value: str) -> float:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except (ValueError, OverflowError, OSError):
                return 0.0

        def truncate(text: str, size: int) -> str:
            marker = "…[已截断]"
            return text if len(text) <= size else text[: max(0, size - len(marker))] + marker[:size]

        async with self._conversation_lock:
            async with self._sessions_lock:
                sessions = await asyncio.to_thread(self.load)
            candidates, cursors = [], {}
            for session_index, entry in enumerate(sessions):
                snapshot = await self.runtime.agent_wrapper.read_session(entry.session_id)
                positions = {message.id: index for index, message in enumerate(snapshot.messages)}
                if entry.last_read_message_id is not None and entry.last_read_message_id not in positions:
                    raise ValueError(f"Conversation cursor is no longer present in session {entry.session_id}")
                last_position = positions.get(entry.last_read_message_id, -1)
                turns = []
                for index, message in enumerate(snapshot.messages):
                    if message.source != "chat" or not message.content.strip():
                        continue
                    if message.role == "user" or not turns:
                        turns.append([])
                    turns[-1].append((index, message))
                for turn_index, turn in enumerate(turns):
                    if turn[-1][0] <= last_position:
                        continue
                    if any(not message.completed for _, message in turn):
                        break
                    if snapshot.state.status == "active" and turn_index == len(turns) - 1:
                        break
                    messages = [message for _, message in turn]
                    created_at = next((message.created_at for message in messages if message.created_at), None)
                    candidates.append(
                        (
                            timestamp(created_at or entry.last_active_at),
                            session_index,
                            turn_index,
                            entry.session_id,
                            messages,
                        ),
                    )
                    cursors[entry.session_id] = messages[-1].id
                candidates = sorted(candidates, key=lambda item: item[:3])[-max_turns:]
            chunks, remaining = [], max_chars
            for _time, _session_index, _turn_index, session_id, messages in reversed(candidates):
                headers = [
                    f"[{message.created_at or '时间未知'}][session={session_id}][{message.role}]\n"
                    for message in messages
                ]
                separator_size = 2 * (len(messages) - 1) + (2 if chunks else 0)
                budget = min(max_message_chars, (remaining - sum(map(len, headers)) - separator_size) // len(messages))
                if budget < 1:
                    break
                text = "\n\n".join(
                    header + truncate(message.content.strip(), budget) for header, message in zip(headers, messages)
                )
                chunks.append(text)
                remaining -= len(text) + (2 if len(chunks) > 1 else 0)
            if cursors and not chunks:
                raise ValueError("max_chars is too small for a timestamped conversation round")
            result = "\n\n".join(reversed(chunks))
            if cursors:
                await self.runtime.checkpoint_proactive(round_id, "conversation", result)
                async with self._sessions_lock:
                    current = await asyncio.to_thread(self.load)
                    for entry in current:
                        if entry.session_id in cursors:
                            entry.last_read_message_id = cursors[entry.session_id]
                    await complete_in_thread(
                        self.runtime.files.write_json,
                        self.runtime.session_ids_path,
                        [entry.model_dump() for entry in current],
                    )
            return result

    @asynccontextmanager
    async def session_operation(self):
        """Reject overlapping chat operations and publish activity changes."""
        if self.runtime.session_lock.locked():
            raise RuntimeError("PA is busy; wait for the current operation to finish")
        try:
            async with self.runtime.session_lock:
                self.runtime.event_stream.publish_status()
                yield
        finally:
            self.runtime.event_stream.publish_status()

    @asynccontextmanager
    async def chat_session(self, **reply_kwargs):
        """Select or create the application-owned conversation for one request."""
        async with self.session_operation():
            session_id = self.runtime.session_id
            if session_id is None:
                session_id = await finish_cleanup(self._new_session(reply_kwargs))
            else:
                await self.remember_session(session_id)
            yield session_id

    async def new_session(self, **reply_kwargs) -> str:
        """Create and select an empty conversation without model generation."""
        async with self.session_operation():
            return await finish_cleanup(self._new_session(reply_kwargs))

    async def select_existing_session(self, session_id: str) -> str:
        """Select a readable source conversation without overlapping Main work."""
        async with self.session_operation():
            await self.runtime.agent_wrapper.read_session(session_id)

            async def commit():
                await self.remember_session(session_id)
                self.runtime.session_id = session_id
                self.runtime.event_stream.select_session()

            await finish_cleanup(commit())
        return session_id

    async def _new_session(self, reply_kwargs: dict) -> str:
        session_id = await self.runtime.agent_wrapper.create_session(**reply_kwargs)
        previous = self.runtime.session_id
        try:
            await self.remember_session(session_id)
            self.runtime.session_id = session_id
            self.runtime.event_stream.select_session()
        except BaseException:
            try:
                await self.runtime.agent_wrapper.release_session(session_id)
            except Exception as exc:
                self.runtime.logger.warning(f"Failed to release unselected PA session {session_id}: {exc}")
            raise
        if previous is not None:
            try:
                await self.runtime.agent_wrapper.release_session(previous)
            except Exception as exc:
                self.runtime.logger.warning(f"Failed to release previous PA session {previous}: {exc}")
        return session_id

    async def get_selected_session(self) -> tuple[dict, str | None]:
        """Report state without queueing behind chat generation."""
        while True:
            session_id = self.runtime.session_id
            status, error = None, None
            if self.runtime.session_lock.locked():
                status = "active"
            elif session_id is not None:
                try:
                    status = (await self.runtime.agent_wrapper.get_session_status(session_id)).status
                except Exception as exc:
                    status, error = "error", str(exc)
                if session_id != self.runtime.session_id:
                    continue
                if self.runtime.session_lock.locked():
                    status, error = "active", None
            return {"session_id": session_id, "status": status}, error
