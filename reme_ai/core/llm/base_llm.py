"""Base LLM interface for ReMe.

This module provides an abstract base class for all LLM implementations in ReMe.
It defines the standard interface for streaming and non-streaming chat completions,
with built-in retry logic, error handling, and support for tool calling and reasoning content.
"""

import asyncio
import json
import time
from abc import ABC
from typing import List, Callable, Generator, AsyncGenerator, Any, Optional, Dict

from loguru import logger

from ..enumeration import ChunkEnum, Role
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations.

    This class defines the standard interface for interacting with language models,
    supporting both streaming and non-streaming operations, tool calling, and
    reasoning content extraction. All concrete LLM implementations should inherit
    from this class and implement the required abstract methods.

    The class provides built-in retry logic and error handling for robust operation
    in production environments.
    
    Attributes:
        model_name: Name of the LLM model to use
        max_retries: Maximum number of retries on failure
        raise_exception: Whether to raise exception on final failure
        kwargs: Additional parameters passed to the LLM API
    """

    def __init__(self, model_name: str, max_retries: int = 3, raise_exception: bool = False, **kwargs):
        """
        Initialize the BaseLLM.

        Args:
            model_name: Name of the LLM model to use
            max_retries: Maximum number of retries on failure (default: 3)
            raise_exception: Whether to raise exception on final failure (default: False)
            **kwargs: Additional parameters passed to the LLM API
        """
        self.model_name: str = model_name
        self.max_retries: int = max_retries
        self.raise_exception: bool = raise_exception
        self.kwargs: dict = kwargs

    @staticmethod
    def _process_stream_chunk(
            stream_chunk: StreamChunk,
            state: dict,
            enable_stream_print: bool = False
    ) -> None:
        """
        Process a single stream chunk and update the aggregation state.

        Args:
            stream_chunk: The stream chunk to process
            state: Dictionary containing aggregation state (reasoning_content, answer_content, tool_calls, flags)
            enable_stream_print: Whether to print streaming response to console
        """
        if stream_chunk.chunk_type is ChunkEnum.USAGE:
            if enable_stream_print:
                print(f"\n<usage>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</usage>", flush=True)

        elif stream_chunk.chunk_type is ChunkEnum.THINK:
            if enable_stream_print:
                if not state['enter_think']:
                    state['enter_think'] = True
                    print("<think>\n", end="", flush=True)
                print(stream_chunk.chunk, end="", flush=True)
            state['reasoning_content'] += stream_chunk.chunk

        elif stream_chunk.chunk_type is ChunkEnum.ANSWER:
            if enable_stream_print:
                if not state['enter_answer']:
                    state['enter_answer'] = True
                    if state['enter_think']:
                        print("\n</think>", flush=True)
                print(stream_chunk.chunk, end="", flush=True)
            state['answer_content'] += stream_chunk.chunk

        elif stream_chunk.chunk_type is ChunkEnum.TOOL:
            if enable_stream_print:
                print(f"\n<tool>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</tool>", flush=True)
            state['tool_calls'].append(stream_chunk.chunk)

        elif stream_chunk.chunk_type is ChunkEnum.ERROR:
            if enable_stream_print:
                print(f"\n<error>{stream_chunk.chunk}</error>", flush=True)

    @staticmethod
    def _create_message_from_state(state: dict) -> Message:
        """
        Create a Message object from the aggregation state.

        Args:
            state: Dictionary containing aggregation state

        Returns:
            Complete Message object
        """
        return Message(
            role=Role.ASSISTANT,
            reasoning_content=state['reasoning_content'],
            content=state['answer_content'],
            tool_calls=state['tool_calls'],
        )

    @staticmethod
    def _accumulate_tool_call_chunk(
            tool_call,
            ret_tools: List[ToolCall]
    ) -> None:
        """
        Process and accumulate tool call chunks from streaming response.
        
        OpenAI/LiteLLM stream tool calls incrementally, sending id, name, and 
        arguments in separate chunks. This method accumulates these fragments
        into complete ToolCall objects.
        
        Args:
            tool_call: Tool call chunk from the API response (has index, id, function)
            ret_tools: List to accumulate tool calls (modified in place)
        """
        index = tool_call.index

        # Ensure we have a ToolCall object at this index
        while len(ret_tools) <= index:
            ret_tools.append(ToolCall(index=index))

        # Accumulate tool call parts (id, name, arguments)
        if tool_call.id:
            ret_tools[index].id += tool_call.id

        if tool_call.function and tool_call.function.name:
            ret_tools[index].name += tool_call.function.name

        if tool_call.function and tool_call.function.arguments:
            ret_tools[index].arguments += tool_call.function.arguments

    @staticmethod
    def _validate_and_serialize_tools(
            ret_tools: List[ToolCall],
            tools: Optional[List[ToolCall]]
    ) -> List[Dict]:
        """
        Validate accumulated tool calls and return serialized versions.
        
        This method:
        1. Filters out tools not in the original tool list
        2. Validates that tool arguments are valid JSON
        3. Serializes valid tools for output
        
        Args:
            ret_tools: Accumulated tool calls from stream
            tools: Original tool definitions provided to the model
        
        Returns:
            List of validated and serialized tool call dictionaries
        
        Raises:
            ValueError: If a tool call has invalid JSON arguments
        """
        if not ret_tools:
            return []

        # Create lookup dict for tool validation
        tool_dict: Dict[str, ToolCall] = {x.name: x for x in tools} if tools else {}
        validated_tools = []

        for tool in ret_tools:
            # Skip tools that weren't in the provided tool list
            if tool.name not in tool_dict:
                continue

            # Validate tool arguments are valid JSON
            if not tool.check_argument():
                raise ValueError(
                    f"Tool call {tool.name} has invalid JSON arguments: {tool.arguments}"
                )

            validated_tools.append(tool.simple_output_dump())

        return validated_tools

    def _build_stream_kwargs(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            log_params: bool = True,
            **kwargs
    ) -> dict:
        """
        Build kwargs for streaming chat API calls.

        This method constructs the parameters dict that will be passed to the
        underlying LLM API. Subclasses should override this method to customize
        the parameter construction logic for their specific API requirements.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            log_params: Whether to log the constructed parameters (default: True)
            **kwargs: Additional parameters to merge into the result

        Returns:
            Dictionary of parameters to be passed to the LLM API
        """
        raise NotImplementedError

    async def _stream_chat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            stream_kwargs: Optional[dict] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Internal async method to stream chat completions from the LLM.

        This is an abstract method that must be implemented by subclasses.
        It should yield StreamChunk objects containing different types of
        content (thinking, answer, tool calls, usage, errors) asynchronously.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            stream_kwargs: Dictionary of additional parameters for the API call

        Yields:
            StreamChunk objects containing chunks of the response

        Raises:
            Exception: Any exception raised by the underlying LLM API
        """
        raise NotImplementedError

    def _stream_chat_sync(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            stream_kwargs: Optional[dict] = None) -> Generator[StreamChunk, None, None]:
        """
        Internal sync method to stream chat completions from the LLM.

        This is an abstract method that must be implemented by subclasses.
        It should yield StreamChunk objects containing different types of
        content (thinking, answer, tool calls, usage, errors).

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            stream_kwargs: Dictionary of additional parameters for the API call

        Yields:
            StreamChunk objects containing chunks of the response

        Raises:
            Exception: Any exception raised by the underlying LLM API
        """
        raise NotImplementedError

    async def _stream_with_retry(
            self,
            operation_name: str,
            messages: List[Message],
            tools: Optional[List[ToolCall]],
            stream_kwargs: dict,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Execute async streaming operation with retry logic and error handling.

        Args:
            operation_name: Name of the operation for logging
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            stream_kwargs: Dictionary of parameters for the streaming API call

        Yields:
            StreamChunk objects from the streaming operation or error chunks
        """
        for i in range(self.max_retries):
            try:
                async for chunk in self._stream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
                    yield chunk
                return

            except Exception as e:
                logger.exception(f"{operation_name} with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    return

                yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                await asyncio.sleep(i + 1)

    def _stream_with_retry_sync(
            self,
            operation_name: str,
            messages: List[Message],
            tools: Optional[List[ToolCall]],
            stream_kwargs: dict,
    ) -> Generator[StreamChunk, None, None]:
        """
        Execute a sync streaming operation with retry logic and error handling.

        Args:
            operation_name: Name of the operation for logging
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            stream_kwargs: Dictionary of parameters for the streaming API call

        Yields:
            StreamChunk objects from the streaming operation or error chunks
        """
        for i in range(self.max_retries):
            try:
                for chunk in self._stream_chat_sync(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
                    yield chunk
                return

            except Exception as e:
                logger.exception(f"{operation_name} with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    return

                yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                time.sleep(i + 1)

    async def stream_chat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream chat completions with automatic retry logic and error handling.

        This method wraps _stream_chat with retry logic and error handling.
        It will automatically retry on failures up to max_retries times, with
        exponential backoff between retries.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            **kwargs: Additional parameters for the API call

        Yields:
            StreamChunk objects containing chunks of the response, or error chunks
            if all retries are exhausted and raise_exception is False
        """
        stream_kwargs = self._build_stream_kwargs(messages, tools, **kwargs)
        async for chunk in self._stream_with_retry("stream chat", messages, tools, stream_kwargs):
            yield chunk

    def stream_chat_sync(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """
        Sync stream chat completions with automatic retry logic and error handling.

        This method wraps _stream_chat_sync with retry logic and error handling.
        It will automatically retry on failures up to max_retries times, with
        exponential backoff between retries.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            **kwargs: Additional parameters for the API call

        Yields:
            StreamChunk objects containing chunks of the response, or error chunks
            if all retries are exhausted and raise_exception is False
        """
        stream_kwargs = self._build_stream_kwargs(messages, tools, **kwargs)
        yield from self._stream_with_retry_sync("stream chat sync", messages, tools, stream_kwargs)

    async def _chat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            enable_stream_print: bool = False,
            **kwargs,
    ) -> Message:
        """
        Internal async method to perform a single chat completion by aggregating streaming chunks.

        This method consumes the entire async streaming response from stream_chat()
        and combines all chunks into a single Message object. It separates
        reasoning content, regular answer content, and tool calls.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            enable_stream_print: Whether to print streaming response to console
            **kwargs: Additional parameters for the API call

        Returns:
            Complete Message with all content aggregated from streaming chunks
        """
        state = {
            'enter_think': False,
            'enter_answer': False,
            'reasoning_content': '',
            'answer_content': '',
            'tool_calls': []
        }

        stream_kwargs = self._build_stream_kwargs(messages, tools, **kwargs)
        async for stream_chunk in self._stream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
            self._process_stream_chunk(stream_chunk, state, enable_stream_print)

        return self._create_message_from_state(state)

    def _chat_sync(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            enable_stream_print: bool = False,
            **kwargs,
    ) -> Message:
        """
        Internal sync method to perform a single chat completion by aggregating streaming chunks.

        This method consumes the entire streaming response from stream_chat_sync()
        and combines all chunks into a single Message object. It separates
        reasoning content, regular answer content, and tool calls.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            enable_stream_print: Whether to print streaming response to console
            **kwargs: Additional parameters for the API call

        Returns:
            Complete Message with all content aggregated from streaming chunks
        """
        state = {
            'enter_think': False,
            'enter_answer': False,
            'reasoning_content': '',
            'answer_content': '',
            'tool_calls': []
        }

        stream_kwargs = self._build_stream_kwargs(messages, tools, **kwargs)
        for stream_chunk in self._stream_chat_sync(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
            self._process_stream_chunk(stream_chunk, state, enable_stream_print)

        return self._create_message_from_state(state)

    async def _execute_with_retry(
            self,
            operation_name: str,
            operation_fn: Callable[[], Any],
            callback_fn: Optional[Callable[[Message], Any]] = None,
            default_value: Any = None,
    ) -> Message | Any:
        """
        Execute async operation with retry logic and error handling.

        Args:
            operation_name: Name of the operation for logging
            operation_fn: Async function that executes the operation
            callback_fn: Optional callback to process the result
            default_value: Value to return if all retries fail

        Returns:
            Result of operation_fn or callback_fn(result) if callback is provided
        """
        for i in range(self.max_retries):
            try:
                result = await operation_fn()
                return callback_fn(result) if callback_fn else result

            except Exception as e:
                logger.exception(f"{operation_name} with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    return default_value

                await asyncio.sleep(1 + i)
        return default_value

    def _execute_with_retry_sync(
            self,
            operation_name: str,
            operation_fn: Callable[[], Message],
            callback_fn: Optional[Callable[[Message], Any]] = None,
            default_value: Any = None,
    ) -> Message | Any:
        """
        Execute a sync operation with retry logic and error handling.

        Args:
            operation_name: Name of the operation for logging
            operation_fn: Function that executes the operation
            callback_fn: Optional callback to process the result
            default_value: Value to return if all retries fail

        Returns:
            Result of operation_fn or callback_fn(result) if callback is provided
        """
        for i in range(self.max_retries):
            try:
                result = operation_fn()
                return callback_fn(result) if callback_fn else result

            except Exception as e:
                logger.exception(f"{operation_name} with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    return default_value

                time.sleep(1 + i)
        return default_value

    async def chat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            enable_stream_print: bool = False,
            callback_fn: Optional[Callable[[Message], Any]] = None,
            default_value: Any = None,
            **kwargs,
    ) -> Message | Any:
        """
        Chat completion with automatic retry logic and error handling.

        This method wraps _chat with retry logic and error handling. It will
        automatically retry on failures up to max_retries times, with exponential
        backoff between retries. Optionally, a callback function can be provided
        to process the resulting Message before returning.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            enable_stream_print: Whether to print streaming response to console
            callback_fn: Optional callback to process the Message before returning
            default_value: Value to return if all retries fail and raise_exception is False
            **kwargs: Additional parameters for the API call

        Returns:
            If callback_fn is provided, returns the result of callback_fn(message).
            Otherwise, returns the complete Message object.
            Returns default_value if all retries fail and raise_exception is False.

        Raises:
            Exception: Any exception from the LLM API if raise_exception is True
                      and all retries are exhausted
        """
        return await self._execute_with_retry(
            operation_name="chat",
            operation_fn=lambda: self._chat(
                messages=messages,
                tools=tools,
                enable_stream_print=enable_stream_print,
                **kwargs,
            ),
            callback_fn=callback_fn,
            default_value=default_value,
        )

    def chat_sync(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            enable_stream_print: bool = False,
            callback_fn: Optional[Callable[[Message], Any]] = None,
            default_value: Any = None,
            **kwargs,
    ) -> Message | Any:
        """
        Perform a sync chat completion with automatic retry logic and error handling.

        This method wraps _chat_sync with retry logic and error handling. It will
        automatically retry on failures up to max_retries times, with exponential
        backoff between retries. Optionally, a callback function can be provided
        to process the resulting Message before returning.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            enable_stream_print: Whether to print streaming response to console
            callback_fn: Optional callback to process the Message before returning
            default_value: Value to return if all retries fail and raise_exception is False
            **kwargs: Additional parameters for the API call

        Returns:
            If callback_fn is provided, returns the result of callback_fn(message).
            Otherwise, returns the complete Message object.
            Returns default_value if all retries fail and raise_exception is False.

        Raises:
            Exception: Any exception from the LLM API if raise_exception is True
                      and all retries are exhausted
        """
        return self._execute_with_retry_sync(
            operation_name="chat sync",
            operation_fn=lambda: self._chat_sync(
                messages=messages,
                tools=tools,
                enable_stream_print=enable_stream_print,
                **kwargs,
            ),
            callback_fn=callback_fn,
            default_value=default_value,
        )

    async def close(self):
        """
        Asynchronously close the client connection or clean up resources.

        This async method should be called when the LLM instance is no longer needed
        to properly release any held resources (e.g., HTTP connections, file handles).
        Subclasses should override this method if they need to perform async cleanup.
        """

    def close_sync(self):
        """
        Close the client connection or clean up resources synchronously.

        This method should be called when the LLM instance is no longer needed
        to properly release any held resources (e.g., HTTP connections, file handles).
        Subclasses should override this method if they need to perform cleanup.
        """
