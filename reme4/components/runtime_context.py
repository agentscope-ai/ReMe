"""Per-request runtime context shared across steps and jobs."""

import asyncio

from ..enumeration import ChunkEnum
from ..schema import Response, StreamChunk


class RuntimeContext:
    """Scratch space for a single execution.

    Holds the response object, an optional stream queue, and a free-form
    data dict accessed via mapping-style operators.
    """

    def __init__(
        self,
        response: Response | None = None,
        stream_queue: asyncio.Queue | None = None,
        **kwargs,
    ):
        self.response: Response = response or Response()
        self.stream_queue: asyncio.Queue | None = stream_queue
        self.data: dict = kwargs

    def get(self, key: str, default=None):
        """Get a value from the data dict."""
        return self.data.get(key, default)

    def update(self, data: dict) -> "RuntimeContext":
        """Merge data into the context."""
        self.data.update(data)
        return self

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value):
        self.data[key] = value

    def __delitem__(self, key: str):
        del self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    @property
    def stream(self) -> bool:
        """Whether streaming is enabled."""
        return self.stream_queue is not None

    @classmethod
    def from_context(cls, context: "RuntimeContext | None" = None, **kwargs) -> "RuntimeContext":
        """Reuse or create a RuntimeContext."""
        # Reuse the existing context (merging kwargs) or create a new one.
        if context is None:
            return cls(**kwargs)
        context.update(kwargs)
        return context

    async def _enqueue(self, chunk: StreamChunk) -> None:
        """Put a chunk on the stream queue."""
        if self.stream_queue is None:
            raise RuntimeError("Stream queue not initialized")
        await self.stream_queue.put(chunk)

    async def add_stream_string(self, chunk: str, chunk_type: ChunkEnum) -> "RuntimeContext":
        """Emit a text chunk to the stream queue."""
        # Emit a text chunk to the stream queue.
        await self._enqueue(StreamChunk(chunk_type=chunk_type, chunk=chunk))
        return self

    async def add_stream_done(self) -> "RuntimeContext":
        """Emit the terminal DONE marker to close the stream."""
        # Emit the terminal DONE marker to close the stream.
        await self._enqueue(StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True))
        return self

    def apply_mapping(self, mapping: dict[str, str]) -> "RuntimeContext":
        """Copy data[source] into data[target] for each mapping pair."""
        # Copy data[source] into data[target] for each {source: target} pair.
        if not mapping:
            return self
        for source, target in mapping.items():
            if source in self.data:
                self.data[target] = self.data[source]
        return self
