"""Streaming demo steps: step1 prepares text, step2 streams it char-by-char."""

import asyncio

from ..base_step import BaseStep
from ...components import R
from ...enumeration import ChunkEnum


@R.register("stream_demo_step1")
class StreamDemoStep1(BaseStep):
    """Read query from context, repeat it 10x, write back for Step2 to stream."""

    async def execute(self):
        assert self.context is not None
        query = self.context.get("query", "")
        repeat = int(self.context.get("repeat", 10))

        stream_text = (query * repeat) if query else ""

        self.logger.info(f"[{self.name}] query={query!r}, repeat={repeat}, len={len(stream_text)}")

        self.context["stream_text"] = stream_text
        return self.context.response


@R.register("stream_demo_step2")
class StreamDemoStep2(BaseStep):
    """Stream stream_text char-by-char as CONTENT chunks with 0.1s pacing."""

    async def execute(self):
        assert self.context is not None
        stream_text: str = self.context.get("stream_text", "")
        interval = float(self.context.get("interval", 0.1))

        self.logger.info(f"[{self.name}] streaming {len(stream_text)} chars, interval={interval}s")

        for ch in stream_text:
            await self.context.add_stream_string(ch, ChunkEnum.CONTENT)
            await asyncio.sleep(interval)

        return self.context.response
