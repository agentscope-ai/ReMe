"""Runtime context for managing response states and asynchronous data streaming."""

import asyncio

from .application_context import ApplicationContext
from ..enumeration import ChunkEnum
from ..schema import Response, StreamChunk


class RuntimeContext:
    """Context for execution state, response metadata, and stream queues."""

    def __init__(self, **kwargs):
        """Initialize the context with all keyword arguments stored in data."""
        self.data: dict = kwargs

    def get(self, key: str, default=None):
        """Get a value from data by key, with optional default."""
        return self.data.get(key, default)

    def set(self, key: str, value) -> "RuntimeContext":
        """Set a value in data by key."""
        self.data[key] = value
        return self

    def delete(self, key: str) -> "RuntimeContext":
        """Delete a key from data."""
        if key in self.data:
            del self.data[key]
        return self

    def contains(self, key: str) -> bool:
        """Check if a key exists in data."""
        return key in self.data

    def update(self, data: dict) -> "RuntimeContext":
        """Update data with a dictionary."""
        self.data.update(data)
        return self

    def keys(self) -> list[str]:
        """Get all keys in data."""
        return list(self.data.keys())

    def values(self) -> list:
        """Get all values in data."""
        return list(self.data.values())

    def items(self) -> list[tuple]:
        """Get all key-value pairs in data."""
        return list(self.data.items())

    def __getitem__(self, key: str):
        """Get a value using bracket syntax."""
        return self.data[key]

    def __setitem__(self, key: str, value):
        """Set a value using bracket syntax."""
        self.data[key] = value

    def __delitem__(self, key: str):
        """Delete a key using bracket syntax."""
        del self.data[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key exists using 'in' operator."""
        return key in self.data

    @property
    def response(self) -> Response:
        """Get or create the response object."""
        return self.data.setdefault("response", Response())

    @property
    def stream_queue(self) -> asyncio.Queue:
        """Get the stream queue."""
        return self.data["stream_queue"]

    @property
    def application_context(self) -> ApplicationContext:
        """Get the application context."""
        return self.data["application_context"]

    @classmethod
    def from_context(cls, context: "RuntimeContext | None" = None, **kwargs) -> "RuntimeContext":
        """Create a new context from an existing instance or keywords."""
        if context is None:
            return cls(**kwargs)
        context.data.update(kwargs)
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
            if source in self.data:
                self.data[target] = self.data[source]
        return self

    def validate_required_keys(
        self,
        required_keys: dict[str, bool],
        context_name: str = "context",
    ) -> "RuntimeContext":
        """Ensure all required keys are present in the context.

        Args:
            required_keys: Dictionary mapping key names to boolean indicating if required
            context_name: Name of the context for error messages (e.g., operator name)
        """
        for key, is_required in required_keys.items():
            if is_required and key not in self.data:
                raise ValueError(f"{context_name}: missing required input '{key}'")
        return self
