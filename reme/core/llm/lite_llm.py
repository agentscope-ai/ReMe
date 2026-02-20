"""LiteLLM asynchronous implementation for ReMe."""

from typing import AsyncGenerator

from loguru import logger

from .base_llm import BaseLLM
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


class LiteLLM(BaseLLM):
    """Async LLM implementation using LiteLLM to support multiple providers."""

    def __init__(self, custom_llm_provider: str = "openai", **kwargs):
        """Initialize the LiteLLM client with API configuration and provider settings."""
        super().__init__(**kwargs)
        self.custom_llm_provider: str = custom_llm_provider

    def _build_stream_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        log_params: bool = True,
        model_name: str | None = None,
        **kwargs,
    ) -> dict:
        """Construct and log the parameters dictionary for LiteLLM API calls.

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
            "custom_llm_provider": self.custom_llm_provider,
            **self.kwargs,
            **kwargs,
        }

        # Add API key and base URL if provided
        if self.api_key:
            llm_kwargs["api_key"] = self.api_key
        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Log parameters for debugging, with message/tool counts instead of full content
        if log_params:
            log_kwargs: dict = {}
            for k, v in llm_kwargs.items():
                if k in ["messages", "tools"]:
                    log_kwargs[k] = len(v) if v is not None else 0
                elif k == "api_key":
                    # Mask API key in logs for security
                    log_kwargs[k] = "***" if v else None
                else:
                    log_kwargs[k] = v
            logger.info(f"llm_kwargs={log_kwargs}")

        return llm_kwargs

    async def _stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        stream_kwargs: dict | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Execute async streaming chat requests and yield processed response chunks."""
        import litellm

        stream_kwargs = stream_kwargs or {}
        completion = await litellm.acompletion(**stream_kwargs)
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

            if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                for tool_call in delta.tool_calls:
                    self._accumulate_tool_call_chunk(tool_call, ret_tool_calls)

        for tool_data in self._validate_and_serialize_tools(ret_tool_calls, tools):
            yield StreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool_data)
