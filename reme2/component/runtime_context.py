"""Runtime context for managing response states and asynchronous data streaming."""

import asyncio

from ..enumeration import ChunkEnum
from ..schema import Response, StreamChunk


class RuntimeContext:
    """Context for execution state, response metadata, and stream queues."""

    def __init__(self, **kwargs):
        self.data: dict = kwargs

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def update(self, data: dict) -> "RuntimeContext":
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
    def response(self) -> Response:
        return self.data.setdefault("response", Response())

    @property
    def stream_queue(self) -> asyncio.Queue:
        return self.data["stream_queue"]

    @classmethod
    def from_context(cls, context: "RuntimeContext | None" = None, **kwargs) -> "RuntimeContext":
        if context is None:
            return cls(**kwargs)
        context.update(kwargs)
        return context

    async def _enqueue(self, chunk: StreamChunk) -> None:
        if self.stream_queue:
            await self.stream_queue.put(chunk)

    async def add_stream_string(self, chunk: str, chunk_type: ChunkEnum) -> "RuntimeContext":
        await self._enqueue(StreamChunk(chunk_type=chunk_type, chunk=chunk))
        return self

    async def add_stream_done(self) -> "RuntimeContext":
        await self._enqueue(StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True))
        return self

    def apply_mapping(self, mapping: dict[str, str]) -> "RuntimeContext":
        if not mapping:
            return self
        for source, target in mapping.items():
            if source in self.data:
                self.data[target] = self.data[source]
        return self
