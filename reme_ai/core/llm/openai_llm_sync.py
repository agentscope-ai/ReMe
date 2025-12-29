"""OpenAI sync LLM implementation for ReMe.

This module provides a synchronous OpenAI-compatible LLM implementation that supports
streaming chat completions, tool calling, and reasoning content (for models
that support it, like o1 series). It can work with OpenAI API or any
compatible API endpoints.

For asynchronous operations, use OpenAILLM from openai_llm module.
"""

from typing import List, Generator, Optional

from openai import OpenAI

from .openai_llm import OpenAILLM
from ..context import C
from ..enumeration import ChunkEnum
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


@C.register_llm("openai_sync")
class OpenAILLMSync(OpenAILLM):
    """
    OpenAI-compatible sync LLM implementation.
    
    This class inherits from OpenAILLM and provides synchronous versions of
    the streaming methods. It overrides _create_client() to create a synchronous
    OpenAI client instead of async.
    
    For asynchronous operations, use OpenAILLM from openai_llm module.
    
    Attributes:
        _client: Synchronous OpenAI client instance
    
    Example:
        >>> llm = OpenAILLMSync(
        ...     model_name="gpt-4",
        ...     api_key="sk-...",
        ...     temperature=0.7
        ... )
        >>> messages = [Message(role=Role.USER, content="Hello!")]
        >>> response = llm.chat(messages)
    """

    def _create_client(self):
        """
        Create and return the synchronous OpenAI client instance.
        
        This method overrides the parent class method to provide a synchronous
        client instead of an asynchronous one.
        
        Returns:
            OpenAI client instance for sync operations
        """
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _stream_chat_sync(
        self,
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        stream_kwargs: Optional[dict] = None
    ) -> Generator[StreamChunk, None, None]:
        """
        Internal sync method to stream chat completions from OpenAI API.
        
        This method creates a streaming chat completion request to the OpenAI API
        and processes the response chunks synchronously. It handles three types of content:
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
        # Create streaming completion request to OpenAI API
        stream_kwargs = stream_kwargs or {}
        completion = self._client.chat.completions.create(**stream_kwargs)

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
        """
        Close the synchronous OpenAI client and release resources.
        
        This method properly closes the HTTP connection pool used by the
        synchronous OpenAI client. It should be called when the LLM instance
        is no longer needed to avoid resource leaks.
        
        Example:
            >>> llm = OpenAILLMSync(model_name="gpt-4")
            >>> # ... use the llm ...
            >>> llm.close_sync()
        """
        self._client.close()

