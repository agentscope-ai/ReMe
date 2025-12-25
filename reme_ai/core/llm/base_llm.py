"""Base LLM interface for ReMe.

This module provides an abstract base class for all LLM implementations in ReMe.
It defines the standard interface for streaming and non-streaming chat completions,
with built-in retry logic, error handling, and support for tool calling and reasoning content.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import List, Callable, Generator, AsyncGenerator, Any, Optional

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

    @abstractmethod
    def _build_stream_kwargs(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
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
            **kwargs: Additional parameters to merge into the result

        Returns:
            Dictionary of parameters ready to be passed to the LLM API
        """

    @abstractmethod
    def _stream_chat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            stream_kwargs: Optional[dict] = None) -> Generator[StreamChunk, None, None]:
        """
        Internal method to stream chat completions from the LLM.

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

    def stream_chat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            **kwargs,
    ) -> Generator[StreamChunk, None, None]:
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
        stream_kwargs = self._build_stream_kwargs(messages, tools, stream=True, **kwargs)
        for i in range(self.max_retries):
            try:
                for chunk in self._stream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
                    yield chunk

                return

            except Exception as e:
                logger.exception(f"stream chat with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e

                    yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    return

                yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                time.sleep(i + 1)

    @abstractmethod
    async def _astream_chat(
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

    async def astream_chat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Async stream chat completions with automatic retry logic and error handling.

        This method wraps _astream_chat with retry logic and error handling.
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
        stream_kwargs = self._build_stream_kwargs(messages, tools, stream=True, **kwargs)
        for i in range(self.max_retries):
            try:
                async for chunk in self._astream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
                    yield chunk

                return

            except Exception as e:
                logger.exception(f"async stream chat with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e

                    yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    return

                yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                await asyncio.sleep(i + 1)

    def _chat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            enable_stream_print: bool = False,
            **kwargs,
    ) -> Message:
        """
        Internal method to perform a single chat completion by aggregating streaming chunks.

        This method consumes the entire streaming response from stream_chat()
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
        enter_think = False
        enter_answer = False
        reasoning_content = ""
        answer_content = ""
        tool_calls = []

        stream_kwargs = self._build_stream_kwargs(messages, tools, stream=True, **kwargs)
        for stream_chunk in self._stream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
            if stream_chunk.chunk_type is ChunkEnum.USAGE:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    print(f"\n<usage>{json.dumps(chunk, ensure_ascii=False, indent=2)}</usage>", flush=True)

            elif stream_chunk.chunk_type is ChunkEnum.THINK:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    if not enter_think:
                        enter_think = True
                        print("<think>\n", end="", flush=True)
                    print(chunk, end="", flush=True)

                reasoning_content += chunk

            elif stream_chunk.chunk_type is ChunkEnum.ANSWER:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    if not enter_answer:
                        enter_answer = True
                        if enter_think:
                            print("\n</think>", flush=True)
                    print(chunk, end="", flush=True)

                answer_content += chunk

            elif stream_chunk.chunk_type is ChunkEnum.TOOL:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    print(f"\n<tool>{json.dumps(chunk, ensure_ascii=False, indent=2)}</tool>", flush=True)

                tool_calls.append(chunk)

            elif stream_chunk.chunk_type is ChunkEnum.ERROR:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    # Display error information
                    print(f"\n<error>{chunk}</error>", flush=True)

        return Message(
            role=Role.ASSISTANT,
            reasoning_content=reasoning_content,
            content=answer_content,
            tool_calls=tool_calls,
        )

    def chat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            enable_stream_print: bool = False,
            callback_fn: Optional[Callable[[Message], Any]] = None,
            default_value: Any = None,
            **kwargs,
    ) -> Message | Any:
        """
        Perform a chat completion with automatic retry logic and error handling.

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

        for i in range(self.max_retries):
            try:
                message: Message = self._chat(
                    messages=messages,
                    tools=tools,
                    enable_stream_print=enable_stream_print,
                    **kwargs,
                )

                if callback_fn:
                    return callback_fn(message)
                else:
                    return message

            except Exception as e:
                logger.exception(f"chat with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    return default_value

                time.sleep(1 + i)

    async def _achat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            enable_stream_print: bool = False,
            **kwargs,
    ) -> Message:
        """
        Internal async method to perform a single chat completion by aggregating streaming chunks.

        This method consumes the entire async streaming response from astream_chat()
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
        enter_think = False
        enter_answer = False
        reasoning_content = ""
        answer_content = ""
        tool_calls = []

        stream_kwargs = self._build_stream_kwargs(messages, tools, stream=True, **kwargs)
        async for stream_chunk in self._astream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
            if stream_chunk.chunk_type is ChunkEnum.USAGE:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    print(f"\n<usage>{json.dumps(chunk, ensure_ascii=False, indent=2)}</usage>", flush=True)

            elif stream_chunk.chunk_type is ChunkEnum.THINK:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    if not enter_think:
                        enter_think = True
                        print("<think>\n", end="", flush=True)
                    print(chunk, end="", flush=True)

                reasoning_content += chunk

            elif stream_chunk.chunk_type is ChunkEnum.ANSWER:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    if not enter_answer:
                        enter_answer = True
                        if enter_think:
                            print("\n</think>", flush=True)
                    print(chunk, end="", flush=True)

                answer_content += chunk

            elif stream_chunk.chunk_type is ChunkEnum.TOOL:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    print(f"\n<tool>{json.dumps(chunk, ensure_ascii=False, indent=2)}</tool>", flush=True)

                tool_calls.append(chunk)

            elif stream_chunk.chunk_type is ChunkEnum.ERROR:
                chunk = stream_chunk.chunk
                if enable_stream_print:
                    print(f"\n<error>{chunk}</error>", flush=True)

        return Message(
            role=Role.ASSISTANT,
            reasoning_content=reasoning_content,
            content=answer_content,
            tool_calls=tool_calls,
        )

    async def achat(
            self,
            messages: List[Message],
            tools: Optional[List[ToolCall]] = None,
            enable_stream_print: bool = False,
            callback_fn: Optional[Callable[[Message], Any]] = None,
            default_value: Any = None,
            **kwargs,
    ) -> Message | Any:
        """
        Async chat completion with automatic retry logic and error handling.

        This method wraps _achat with retry logic and error handling. It will
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

        for i in range(self.max_retries):
            try:
                message: Message = await self._achat(
                    messages=messages,
                    tools=tools,
                    enable_stream_print=enable_stream_print,
                    **kwargs,
                )

                if callback_fn:
                    return callback_fn(message)
                else:
                    return message

            except Exception as e:
                logger.exception(f"async chat with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    return default_value

                await asyncio.sleep(1 + i)

    def close(self):
        """
        Close the client connection or clean up resources.

        This method should be called when the LLM instance is no longer needed
        to properly release any held resources (e.g., HTTP connections, file handles).
        Subclasses should override this method if they need to perform cleanup.
        """

    async def async_close(self):
        """
        Asynchronously close the client connection or clean up resources.

        This async method should be called when the LLM instance is no longer needed
        to properly release any held resources (e.g., HTTP connections, file handles).
        Subclasses should override this method if they need to perform async cleanup.
        """
