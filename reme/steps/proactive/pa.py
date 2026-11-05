"""Interact with the application's current agent session."""

import asyncio
from contextlib import asynccontextmanager

from ..common.chat import ChatStep
from ..base_step import BaseStep
from ...components import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk, collect_reply
from ...utils.async_utils import finish_cleanup


class _PaStep(BaseStep):
    """Share server-owned chat configuration with an empty prompt unless configured; serialize session changes."""

    @property
    def _state(self) -> dict:
        return self.app_context.metadata.setdefault("pa", {"session_id": None, "lock": asyncio.Lock()})

    def _validate(self, *parameters: str) -> None:
        if self.app_context is None or self.agent_wrapper is None:
            raise RuntimeError("PA jobs require an ApplicationContext and agent_wrapper")
        unexpected = self.context.data.keys() - {*parameters, "metadata"}
        if unexpected:
            raise ValueError(f"Unexpected PA parameters: {', '.join(sorted(unexpected))}")

    def _reply_kwargs(self) -> dict:
        return {
            "system_prompt": self.kwargs.get("system_prompt") or "",
            "job_tools": self.kwargs.get("job_tools", ChatStep.READ_ONLY_TOOLS),
            "builtin_tools": [],
        }

    @asynccontextmanager
    async def _operation(self):
        if self._state["lock"].locked():
            raise RuntimeError("PA is busy; wait for the current operation to finish")
        async with self._state["lock"]:
            yield

    async def _new_session(self) -> str:
        session_id = await self.agent_wrapper.create_session(**self._reply_kwargs())
        previous = self._state["session_id"]
        self._state["session_id"] = session_id
        if previous is not None:
            try:
                await self.agent_wrapper.release_session(previous)
            except Exception as exc:
                self.logger.warning(f"Failed to release previous PA session {previous}: {exc}")
        return session_id


@R.register("pa_chat_step")
class PaChatStep(_PaStep):
    """Stream a user message through the current session, creating it on first use."""

    async def execute(self):
        self._validate("query")
        query = self.context.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        async with self._operation():
            session_id = self._state["session_id"] or await finish_cleanup(self._new_session())
            # Bind the browser to the selected session even if generation fails before reply_start.
            await self.context.add_stream_chunk(
                StreamChunk(
                    chunk_type=ChunkEnum.DATA,
                    chunk=None,
                    session_id=session_id,
                    metadata={"visibility": "diagnostic"},
                ),
            )
            result = await collect_reply(
                self.agent_wrapper.reply_stream(query.strip(), resume=session_id, **self._reply_kwargs()),
                on_chunk=self.context.add_stream_chunk,
            )
            self.context.response.success = result["status"] == "completed" and self.context.stream_error is None
            self.context.response.answer = result["result"]
            self.context.response.metadata.update(result)
        return self.context.response


@R.register("pa_session_step")
class PaSessionStep(_PaStep):
    """Return the current session ID and activity without waiting behind generation."""

    async def execute(self):
        self._validate()
        while True:
            session_id = self._state["session_id"]
            status, error = None, None
            if self._state["lock"].locked():
                status = "active"
            elif session_id is not None:
                try:
                    status = (await self.agent_wrapper.get_session_status(session_id)).status
                except Exception as exc:
                    status, error = "error", str(exc)
                if session_id != self._state["session_id"]:
                    continue
                if self._state["lock"].locked():
                    status, error = "active", None
            break
        self.context.response.answer = {"session_id": session_id, "status": status}
        if error is not None:
            self.context.response.metadata["error"] = error
        return self.context.response


@R.register("pa_new_session_step")
class PaNewSessionStep(_PaStep):
    """Create and select a new empty session, preserving the old selection on failure."""

    async def execute(self):
        self._validate()
        async with self._operation():
            self.context.response.answer = await finish_cleanup(self._new_session())
        return self.context.response
