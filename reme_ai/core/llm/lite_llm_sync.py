"""LiteLLM sync implementation for ReMe.

This module provides a synchronous LiteLLM-based LLM implementation that supports
streaming chat completions, tool calling, and reasoning content. LiteLLM
provides a unified interface to 100+ LLM providers including OpenAI,
Anthropic, Azure, Bedrock, Vertex AI, and more.

For asynchronous operations, use LiteLLM from lite_llm module.
"""

from typing import List, Generator, Optional

import litellm

from .lite_llm import LiteLLM
from ..context import C
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


@C.register_llm("litellm_sync")
class LiteLLMSync(LiteLLM):
    """
    LiteLLM-based sync LLM implementation.
    
    This class inherits from LiteLLM and provides synchronous versions of
    the streaming methods. It reuses the __init__ and _build_stream_kwargs
    methods from the parent class.
    
    For asynchronous operations, use LiteLLM from lite_llm module.
    
    Example:
        >>> llm = LiteLLMSync(
        ...     model_name="gpt-4",
        ...     api_key="sk-...",
        ...     temperature=0.7
        ... )
        >>> messages = [Message(role=Role.USER, content="Hello!")]
        >>> response = llm.chat(messages)
    """

    def _stream_chat_sync(
        self,
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        stream_kwargs: Optional[dict] = None
    ) -> Generator[StreamChunk, None, None]:
        """
        Internal sync method to stream chat completions from LiteLLM API.
        
        This method creates a streaming chat completion request using LiteLLM
        and processes the response chunks synchronously. It handles three types of content:
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
        # Create streaming completion request using LiteLLM
        stream_kwargs = stream_kwargs or {}
        completion = litellm.completion(**stream_kwargs)

        # Track accumulated tool calls across chunks
        ret_tools: List[ToolCall] = []
        # Flag to track if we've started receiving answer content
        is_answering: bool = False

        for chunk in completion:
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

    def close_sync(self):
        """
        Close the LiteLLM client and release resources.
        
        Note: LiteLLM uses a stateless function-based API, so there's no
        persistent client connection to close. This method is provided for
        API consistency with other LLM implementations.
        """
        pass

