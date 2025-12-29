"""LiteLLM async implementation for ReMe.

This module provides an async LiteLLM-based LLM implementation that supports
streaming chat completions, tool calling, and reasoning content. LiteLLM
provides a unified interface to 100+ LLM providers including OpenAI,
Anthropic, Azure, Bedrock, Vertex AI, and more.

For synchronous operations, use LiteLLMSync from lite_llm_sync module.
"""

import os
from typing import List, AsyncGenerator, Optional

import litellm
from loguru import logger

from .base_llm import BaseLLM
from ..context import C
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


@C.register_llm("litellm")
class LiteLLM(BaseLLM):
    """
    LiteLLM-based async LLM implementation.
    
    This class provides async integration with LiteLLM, which offers a unified
    interface to 100+ LLM providers. It supports:
    - Async streaming and non-streaming chat completions
    - Tool/function calling
    - Reasoning content extraction (for models that support thinking)
    - Multiple providers (OpenAI, Anthropic, Azure, Bedrock, etc.)
    
    For synchronous operations, use LiteLLMSync from lite_llm_sync module.
    
    The API key and base URL can be provided either as constructor arguments
    or through environment variables (REME_LLM_API_KEY and REME_LLM_BASE_URL).
    
    Attributes:
        api_key: API key for the LLM provider
        base_url: Optional base URL for custom API endpoints
        custom_llm_provider: LLM provider to use (default: "openai")
    
    Example:
        >>> llm = LiteLLM(
        ...     model_name="qwen3-max",
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
        custom_llm_provider: str = "openai",
        **kwargs
    ):
        """
        Initialize the LiteLLM client.
        
        Args:
            api_key: API key for the provider. If None, reads from REME_LLM_API_KEY env var
                    or provider-specific env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
            base_url: Base URL for API endpoint. If None, reads from REME_LLM_BASE_URL env var
            custom_llm_provider: LLM provider to use (default: "openai"). Supported values include
                                "openai", "anthropic", "azure", "bedrock", "vertex_ai", etc.
            **kwargs: Additional arguments passed to BaseLLM, including:
                - model_name: Name of the model to use (required).
                - max_retries: Maximum retry attempts on failure (default: 3)
                - raise_exception: Whether to raise exception on failure (default: False)
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - top_p: Nucleus sampling parameter
                - And other LiteLLM API parameters
        """
        super().__init__(**kwargs)
        self.api_key: Optional[str] = api_key or os.getenv("REME_LLM_API_KEY")
        self.base_url: Optional[str] = base_url or os.getenv("REME_LLM_BASE_URL")
        self.custom_llm_provider: str = custom_llm_provider
    
    def _build_stream_kwargs(
        self,
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        log_params: bool = True,
        **kwargs
    ) -> dict:
        """
        Build kwargs for LiteLLM completion API calls.
        
        This method constructs the parameters dictionary for the LiteLLM API,
        combining messages, tools, instance-level kwargs, and call-specific kwargs.
        It also logs the parameters for debugging purposes (excluding full message content).
        
        Args:
            messages: List of conversation messages to send to the model
            tools: Optional list of tool definitions available for the model to call
            log_params: Whether to log the constructed parameters (default: True)
            **kwargs: Additional parameters to pass to the API (e.g., temperature, max_tokens)
        
        Returns:
            Dictionary of parameters for LiteLLM API call
        """
        # Construct the API parameters by merging multiple sources
        llm_kwargs = {
            "model": self.model_name,
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
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        stream_kwargs: Optional[dict] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Internal async method to stream chat completions from LiteLLM API.
        
        This method creates a streaming chat completion request using LiteLLM 
        and processes the response chunks asynchronously. It handles three types of content:
        1. Reasoning content (for models that support thinking)
        2. Regular text responses
        3. Tool/function calls
        
        The method accumulates tool call information across multiple chunks since
        LiteLLM streams tool calls in fragments (id, name, arguments separately).
        
        Args:
            messages: List of conversation messages to send to the model
            tools: Optional list of tool definitions available for the model to call
            stream_kwargs: Dictionary of additional parameters for the LiteLLM API (already built by caller)
        
        Yields:
            StreamChunk objects with different chunk types:
            - USAGE: Token usage information
            - THINK: Reasoning/thinking content (for supported models)
            - ANSWER: Regular text response content
            - TOOL: Complete tool call information
        
        Raises:
            ValueError: If a tool call has invalid arguments
        """
        # Create streaming completion request using LiteLLM asynchronously
        stream_kwargs = stream_kwargs or {}
        completion = await litellm.acompletion(**stream_kwargs)

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

                # Check for reasoning content (models that support thinking)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    yield StreamChunk(chunk_type=ChunkEnum.THINK, chunk=delta.reasoning_content)

                else:
                    if not is_answering:
                        is_answering = True

                    # Yield regular text content
                    if delta.content is not None:
                        yield StreamChunk(chunk_type=ChunkEnum.ANSWER, chunk=delta.content)

                    # Process tool calls - LiteLLM streams them incrementally
                    if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                        for tool_call in delta.tool_calls:
                            self._accumulate_tool_call_chunk(tool_call, ret_tools)

        # After streaming completes, validate and yield complete tool calls
        for tool_data in self._validate_and_serialize_tools(ret_tools, tools):
            yield StreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool_data)


