import asyncio

from .base_context import BaseContext
from ..enumeration import ChunkEnum
from ..schema import Response
from ..schema import StreamChunk


class RuntimeContext(BaseContext):
    def __init__(
            self,
            response: Response | None = None,
            stream_queue: asyncio.Queue | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.response: Response | None = response if response is not None else Response()
        self.stream_queue: asyncio.Queue | None = stream_queue

    async def add_stream_string_and_type(self, chunk: str, chunk_type: ChunkEnum):
        if self.stream_queue is None:
            return self
        stream_chunk = StreamChunk(chunk_type=chunk_type, chunk=chunk)
        await self.stream_queue.put(stream_chunk)
        return self

    async def add_stream_chunk(self, stream_chunk: StreamChunk):
        if self.stream_queue is None:
            return self
        await self.stream_queue.put(stream_chunk)
        return self

    async def add_stream_done(self):
        if self.stream_queue is None:
            return self
        done_chunk = StreamChunk(chunk_type=ChunkEnum.DONE, chunk="", done=True)
        await self.stream_queue.put(done_chunk)
        return self

    def add_response_error(self, e: Exception):
        self.response.success = False
        self.response.answer = str(e.args)
