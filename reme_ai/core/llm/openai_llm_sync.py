"""Synchronous OpenAI-compatible LLM implementation supporting streaming, tool calls, and reasoning content."""

from typing import Generator, Optional

from openai import OpenAI

from .openai_llm import OpenAILLM
from ..context import C
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


@C.register_llm("openai_sync")
class OpenAILLMSync(OpenAILLM):
    """Synchronous LLM client for OpenAI-compatible APIs, inheriting from OpenAILLM."""

    def _create_client(self):
        """Create and return an instance of the synchronous OpenAI client."""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _stream_chat_sync(
        self,
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        stream_kwargs: Optional[dict] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Synchronously generate a stream of chat completion chunks including text, reasoning, and tool calls."""
        # Create streaming completion request to OpenAI API
        stream_kwargs = stream_kwargs or {}
        completion = self._client.chat.completions.create(**stream_kwargs)

        # Track accumulated tool calls across chunks
        ret_tools: list[ToolCall] = []
        # Flag to track if we've started receiving answer content
        is_answering: bool = False

        for chunk in completion:
            # Handle usage information (typically the last chunk)
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    yield StreamChunk(chunk_type=ChunkEnum.USAGE, chunk=chunk.usage.model_dump())

            else:
                delta = chunk.choices[0].delta

                # Check for reasoning content (o1-preview, o1-mini models)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    yield StreamChunk(chunk_type=ChunkEnum.THINK, chunk=delta.reasoning_content)

                else:
                    if not is_answering:
                        is_answering = True

                    # Yield regular text content
                    if delta.content is not None:
                        yield StreamChunk(chunk_type=ChunkEnum.ANSWER, chunk=delta.content)

                    # Process tool calls - OpenAI streams them incrementally
                    if delta.tool_calls is not None:
                        for tool_call in delta.tool_calls:
                            self._accumulate_tool_call_chunk(tool_call, ret_tools)

        # After streaming completes, validate and yield complete tool calls
        for tool_data in self._validate_and_serialize_tools(ret_tools, tools):
            yield StreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool_data)

    def close_sync(self):
        """Close the synchronous OpenAI client and release network resources."""
        self._client.close()
