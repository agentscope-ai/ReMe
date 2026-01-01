"""Runtime context for managing response states and asynchronous data streaming."""

import asyncio

from .base_context import BaseContext
from ..enumeration import ChunkEnum
from ..schema import Response, StreamChunk


class RuntimeContext(BaseContext):
    """Context for execution state, response metadata, and stream queues."""

    def __init__(
        self,
        response: Response | None = None,
        stream_queue: asyncio.Queue | None = None,
        **kwargs,
    ):
        """Initialize the context with optional response and queue."""
        super().__init__(**kwargs)
        self.response = response or Response()
        self.stream_queue = stream_queue

    @classmethod
    def from_context(cls, context: "RuntimeContext | None" = None, **kwargs) -> "RuntimeContext":
        """Create a new context from an existing instance or keywords."""
        if context is None:
            return cls(**kwargs)

        context.update(kwargs)
        return context

    async def _enqueue(self, chunk: StreamChunk) -> None:
        """Internal helper to put a chunk into the queue if it exists."""
        if self.stream_queue:
            await self.stream_queue.put(chunk)

    async def add_stream_string(self, chunk: str, chunk_type: ChunkEnum) -> "RuntimeContext":
        """Enqueue a stream chunk from a raw string and type."""
        await self._enqueue(StreamChunk(chunk_type=chunk_type, chunk=chunk))
        return self

    async def add_stream_chunk(self, stream_chunk: StreamChunk) -> "RuntimeContext":
        """Enqueue an existing stream chunk."""
        await self._enqueue(stream_chunk)
        return self

    async def add_stream_done(self) -> "RuntimeContext":
        """Enqueue a termination chunk to signal the end of the stream."""
        await self._enqueue(StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True))
        return self

    def add_response_error(self, e: Exception) -> "RuntimeContext":
        """Record an exception into the response object."""
        self.response.success = False
        self.response.answer = str(e)
        return self

    def apply_mapping(self, mapping: dict[str, str]) -> "RuntimeContext":
        """Copy internal values based on a source-to-target key map."""
        if not mapping:
            return self

        for source, target in mapping.items():
            if source in self:
                self[target] = self[source]
        return self

    def validate_required_keys(self, required_keys: dict[str, bool], context_name: str = "context") -> "RuntimeContext":
        """Ensure all required keys are present in the context.

        Args:
            required_keys: Dictionary mapping key names to boolean indicating if required
            context_name: Name of the context for error messages (e.g., operator name)
        """
        for key, is_required in required_keys.items():
            if is_required and key not in self:
                raise ValueError(f"{context_name}: missing required input '{key}'")
        return self
