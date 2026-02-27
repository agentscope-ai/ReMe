"""Synchronous OpenAI-compatible LLM implementation supporting streaming, tool calls, and reasoning content."""

from typing import Generator

from openai import OpenAI

from .openai_llm import OpenAILLM
from ..enumeration import ChunkEnum
from ..schema import Message, StreamChunk, ToolCall


class OpenAILLMSync(OpenAILLM):
    """Synchronous LLM client for OpenAI-compatible APIs, inheriting from OpenAILLM."""

    def _create_client(self):
        """Create and return an instance of the synchronous OpenAI client."""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _stream_chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        stream_kwargs: dict | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Synchronously generate a stream of chat completion chunks including text, reasoning, and tool calls."""
        stream_kwargs = stream_kwargs or {}
        completion = self.client.chat.completions.create(**stream_kwargs)
        ret_tool_calls: list[ToolCall] = []

        for chunk in completion:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    yield StreamChunk(chunk_type=ChunkEnum.USAGE, chunk=chunk.usage.model_dump())
                continue

            delta = chunk.choices[0].delta

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                yield StreamChunk(chunk_type=ChunkEnum.THINK, chunk=delta.reasoning_content)

            if delta.content:
                yield StreamChunk(chunk_type=ChunkEnum.ANSWER, chunk=delta.content)

            if delta.tool_calls is not None:
                for tool_call in delta.tool_calls:
                    self._accumulate_tool_call_chunk(tool_call, ret_tool_calls)

        for tool_data in self._validate_and_serialize_tools(ret_tool_calls, tools):
            yield StreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool_data)

    def close_sync(self):
        """Close the synchronous OpenAI client and release network resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
        super().close_sync()
