"""Read-only subscription to the application's current PA conversation."""

from contextlib import aclosing

from .base_pa_step import BasePaStep
from ...components import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk


@R.register("pa_events_step")
class PaEventsStep(BasePaStep):
    """Relay cursor-based changes through the existing StreamJob transport."""

    async def execute(self):
        self._validate("cursor")
        async with aclosing(self.pa.events(self.context.get("cursor"))) as events:
            async for event in events:
                await self.context.add_stream_chunk(
                    StreamChunk(
                        chunk_type=ChunkEnum.DATA,
                        session_id=event.session_id,
                        chunk=event.model_dump(mode="json", exclude_none=True),
                    ),
                )
        return self.context.response
