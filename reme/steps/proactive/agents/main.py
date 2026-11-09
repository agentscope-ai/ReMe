"""Remy's PA conversation; the runtime owns session selection and lifecycle."""

from .base import BasePaAgentStep
from ....components import R
from ....enumeration import ChunkEnum
from ....schema import StreamChunk, collect_reply


@R.register("pa_chat_step")
class PaChatStep(BasePaAgentStep):
    """Stream a user request through the selected native session."""

    async def _reply_kwargs(self) -> dict:
        await self._ensure_memory()
        return self._agent_options(
            injected={"pa_caller": "chat"},
            now=self._now().isoformat(),
            cron_runs_path=self.pa.cron_runs_path,
            cron_results_dir=self.pa.tasks.root_path,
            temporary_results_dir=self.pa.temp_tasks.root_path,
        )

    async def execute(self):
        self._validate("query")
        query = self.context.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        reply_kwargs = await self._reply_kwargs()
        async with self.pa.sessions.chat_session(**reply_kwargs) as session_id:
            await self.context.add_stream_chunk(
                StreamChunk(
                    chunk_type=ChunkEnum.DATA,
                    chunk=None,
                    session_id=session_id,
                    metadata={"visibility": "diagnostic"},
                ),
            )
            result = await collect_reply(
                self.agent_wrapper.reply_stream(query.strip(), resume=session_id, **reply_kwargs),
                on_chunk=self.context.add_stream_chunk,
            )
            self.context.response.success = result["status"] == "completed" and self.context.stream_error is None
            self.context.response.answer = result["result"]
            self.context.response.metadata.update(result)
        return self.context.response


@R.register("pa_new_session_step")
class PaNewSessionStep(PaChatStep):
    """Select an empty chat session using the same persona and tools as chat."""

    async def execute(self):
        self._validate()
        reply_kwargs = await self._reply_kwargs()
        self.context.response.answer = await self.pa.sessions.new_session(**reply_kwargs)
        return self.context.response
