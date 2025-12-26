"""OpenAI async LLM implementation for ReMe.

This module provides an async OpenAI-compatible LLM implementation that supports
streaming chat completions, tool calling, and reasoning content (for models
that support it, like o1 series). It can work with OpenAI API or any
compatible API endpoints.

For synchronous operations, use OpenAILLMSync from openai_llm_sync module.
"""

import os
from typing import List, AsyncGenerator, Optional

from loguru import logger
from openai import AsyncOpenAI

from .base_llm import BaseLLM
from ..context import C
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


@C.register_llm("openai")
class OpenAILLM(BaseLLM):
    """
    OpenAI-compatible async LLM implementation.
    
    This class provides async integration with OpenAI's Chat Completions API or
    any compatible API endpoint (e.g., Azure OpenAI, local models with
    OpenAI-compatible servers). It supports:
    - Async streaming and non-streaming chat completions
    - Tool/function calling
    - Reasoning content extraction (for o1-preview, o1-mini, etc.)
    
    For synchronous operations, use OpenAILLMSync from openai_llm_sync module.
    
    The API key and base URL can be provided either as constructor arguments
    or through environment variables (REME_LLM_API_KEY and REME_LLM_BASE_URL).
    
    Attributes:
        api_key: OpenAI API key or compatible service key
        base_url: Base URL for the API endpoint
        _aclient: Asynchronous OpenAI client instance
    
    Example:
        >>> llm = OpenAILLM(
        ...     model_name="gpt-4",
        ...     api_key="sk-...",
        ...     temperature=0.7
        ... )
        >>> messages = [Message(role=Role.USER, content="Hello!")]
        >>> response = await llm.achat(messages)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI async LLM client.
        
        Args:
            api_key: OpenAI API key. If None, reads from REME_LLM_API_KEY env var
            base_url: Base URL for API endpoint. If None, reads from REME_LLM_BASE_URL env var
            **kwargs: Additional arguments passed to BaseLLM, including:
                - model_name: Name of the model to use (required)
                - max_retries: Maximum retry attempts on failure (default: 3)
                - raise_exception: Whether to raise exception on failure (default: False)
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - top_p: Nucleus sampling parameter
                - And other OpenAI API parameters
        """
        super().__init__(**kwargs)
        self.api_key: str = api_key or os.getenv("REME_LLM_API_KEY", "")
        self.base_url: str = base_url or os.getenv("REME_LLM_BASE_URL", "")
        
        # Create async client
        self._aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def _build_stream_kwargs(
        self,
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        log_params: bool = True,
        **kwargs
    ) -> dict:
        """
        Build kwargs for OpenAI Chat Completions API calls.
        
        This method constructs the parameters dictionary for the OpenAI API,
        combining messages, tools, instance-level kwargs, and call-specific kwargs.
        It also logs the parameters for debugging purposes (excluding full message content).
        
        Args:
            messages: List of conversation messages to send to the model
            tools: Optional list of tool definitions available for the model to call
            log_params: Whether to log the constructed parameters (default: True)
            **kwargs: Additional parameters to pass to the API (e.g., temperature, max_tokens)
        
        Returns:
            Dictionary of parameters ready for OpenAI API call
        """
        # Construct the API parameters by merging multiple sources
        llm_kwargs = {
            "model": self.model_name,
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
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        stream_kwargs: Optional[dict] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Internal async method to stream chat completions from OpenAI API.
        
        This method creates a streaming chat completion request to the OpenAI API 
        and processes the response chunks asynchronously. It handles three types of content:
        1. Reasoning content (for models like o1 that support thinking)
        2. Regular text responses
        3. Tool/function calls
        
        The method accumulates tool call information across multiple chunks since
        OpenAI streams tool calls in fragments (id, name, arguments separately).
        
        Args:
            messages: List of conversation messages to send to the model
            tools: Optional list of tool definitions available for the model to call
            stream_kwargs: Dictionary of additional parameters for the OpenAI API (already built by caller)
        
        Yields:
            StreamChunk objects with different chunk types:
            - USAGE: Token usage information
            - THINK: Reasoning/thinking content (for supported models)
            - ANSWER: Regular text response content
            - TOOL: Complete tool call information
        
        Raises:
            ValueError: If a tool call has invalid arguments
        """
        # Create streaming completion request to OpenAI API asynchronously
        stream_kwargs = stream_kwargs or {}
        completion = await self._aclient.chat.completions.create(**stream_kwargs)

        # Track accumulated tool calls across chunks
        ret_tools: List[ToolCall] = []
        # Flag to track if we've started receiving answer content
        is_answering: bool = False

        async for chunk in completion:
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

    async def close(self):
        """
        Asynchronously close the async OpenAI client and release resources.
        
        This method properly closes the HTTP connection pool used by the
        asynchronous OpenAI client. It should be called when the LLM instance
        is no longer needed to avoid resource leaks.
        
        Example:
            >>> llm = OpenAILLM(model_name="gpt-4")
            >>> # ... use the llm asynchronously ...
            >>> await llm.close()
        """
        await self._aclient.close()
