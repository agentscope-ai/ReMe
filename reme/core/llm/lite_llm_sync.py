"""Synchronous LiteLLM-based LLM implementation for the ReMe framework."""

from typing import Generator

from .lite_llm import LiteLLM
from ..enumeration import ChunkEnum
from ..schema import Message, StreamChunk, ToolCall


class LiteLLMSync(LiteLLM):
    """Synchronous LiteLLM client for executing chat completions and streaming responses."""

    def _stream_chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        stream_kwargs: dict | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Internal synchronous generator for processing streaming chat completion chunks."""
        import litellm

        stream_kwargs = stream_kwargs or {}
        completion = litellm.completion(**stream_kwargs)
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

            if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                for tool_call in delta.tool_calls:
                    self._accumulate_tool_call_chunk(tool_call, ret_tool_calls)

        for tool_data in self._validate_and_serialize_tools(ret_tool_calls, tools):
            yield StreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool_data)
