"""Asynchronous OpenAI-compatible LLM implementation supporting streaming, tool calls, and reasoning content."""

from typing import AsyncGenerator

from loguru import logger
from openai import AsyncOpenAI

from .base_llm import BaseLLM
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


class OpenAILLM(BaseLLM):
    """Asynchronous LLM client for OpenAI-compatible APIs supporting streaming completions and tool execution."""

    def __init__(self, **kwargs):
        """Initialize the OpenAI async client with API credentials and model configuration."""
        super().__init__(**kwargs)

        # Lazy client initialization
        self._client = None

    def _create_client(self):
        """Create and return an instance of the AsyncOpenAI client."""
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @property
    def client(self):
        """Lazily create and return the AsyncOpenAI client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _build_stream_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        log_params: bool = True,
        model_name: str | None = None,
        **kwargs,
    ) -> dict:
        """Construct the parameter dictionary for the OpenAI Chat Completions API call.

        Args:
            messages: List of conversation messages
            tools: Optional list of tool calls
            log_params: Whether to log parameters
            model_name: Optional model name to override self.model_name
            **kwargs: Additional parameters
        """
        # Use the provided model_name or fall back to self.model_name
        effective_model = model_name if model_name is not None else self.model_name

        # Construct the API parameters by merging multiple sources
        llm_kwargs = {
            "model": effective_model,
            "messages": [x.simple_dump() for x in messages],
            "tools": [x.simple_input_dump() for x in tools] if tools else None,
            "stream": True,
            **self.kwargs,
            **kwargs,
        }

        # Log parameters for debugging, with message/tool counts instead of full content
        if log_params:
            log_kwargs: dict = {}
            for k, v in llm_kwargs.items():
                if k in ["messages", "tools"]:
                    log_kwargs[k] = len(v) if v is not None else 0
                else:
                    log_kwargs[k] = v
            logger.info(f"llm_kwargs={log_kwargs}")

        return llm_kwargs

    async def _stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None,
        stream_kwargs: dict,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a stream of chat completion chunks including text, reasoning content, and tool calls."""
        stream_kwargs = stream_kwargs or {}
        completion = await self.client.chat.completions.create(**stream_kwargs)
        ret_tool_calls: list[ToolCall] = []

        async for chunk in completion:
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

    async def close(self):
        """Asynchronously close the OpenAI client and release network resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
        await super().close()
