"""Abstract base interface for ReMe LLM implementations."""

import asyncio
import json
import time
from abc import ABC
from typing import Callable, Generator, AsyncGenerator, Optional, Any

from loguru import logger

from ..enumeration import ChunkEnum, Role
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


class BaseLLM(ABC):
    """Abstract base class defining the standard interface for LLM interactions."""

    def __init__(self, model_name: str, max_retries: int = 3, raise_exception: bool = False, **kwargs):
        """Initialize the LLM client with model configurations and retry policies."""
        self.model_name: str = model_name
        self.max_retries: int = max_retries
        self.raise_exception: bool = raise_exception
        self.kwargs: dict = kwargs

    @staticmethod
    def _process_stream_chunk(
        stream_chunk: StreamChunk,
        state: dict,
        enable_stream_print: bool = False,
    ) -> None:
        """Update the aggregation state by processing an individual stream chunk."""
        if stream_chunk.chunk_type is ChunkEnum.USAGE:
            if enable_stream_print:
                print(f"\n<usage>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</usage>", flush=True)

        elif stream_chunk.chunk_type is ChunkEnum.THINK:
            if enable_stream_print:
                if not state["enter_think"]:
                    state["enter_think"] = True
                    print("<think>\n", end="", flush=True)
                print(stream_chunk.chunk, end="", flush=True)
            state["reasoning_content"] += stream_chunk.chunk

        elif stream_chunk.chunk_type is ChunkEnum.ANSWER:
            if enable_stream_print:
                if not state["enter_answer"]:
                    state["enter_answer"] = True
                    if state["enter_think"]:
                        print("\n</think>", flush=True)
                print(stream_chunk.chunk, end="", flush=True)
            state["answer_content"] += stream_chunk.chunk

        elif stream_chunk.chunk_type is ChunkEnum.TOOL:
            if enable_stream_print:
                print(f"\n<tool>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</tool>", flush=True)
            state["tool_calls"].append(stream_chunk.chunk)

        elif stream_chunk.chunk_type is ChunkEnum.ERROR:
            if enable_stream_print:
                print(f"\n<error>{stream_chunk.chunk}</error>", flush=True)

    @staticmethod
    def _create_message_from_state(state: dict) -> Message:
        """Construct a Message object from the accumulated aggregation state."""
        return Message(
            role=Role.ASSISTANT,
            reasoning_content=state["reasoning_content"],
            content=state["answer_content"],
            tool_calls=state["tool_calls"],
        )

    @staticmethod
    def _accumulate_tool_call_chunk(
        tool_call,
        ret_tools: list[ToolCall],
    ) -> None:
        """Assemble incremental tool call fragments into complete ToolCall objects."""
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
        ret_tools: list[ToolCall],
        tools: Optional[list[ToolCall]],
    ) -> list[dict]:
        """Validate tool call integrity and return serialized tool dictionaries."""
        if not ret_tools:
            return []

        # Create lookup dict for tool validation
        tool_dict: dict[str, ToolCall] = {x.name: x for x in tools} if tools else {}
        validated_tools = []

        for tool in ret_tools:
            # Skip tools that weren't in the provided tool list
            if tool.name not in tool_dict:
                continue

            # Validate tool arguments are valid JSON
            if not tool.check_argument():
                raise ValueError(
                    f"Tool call {tool.name} has invalid JSON arguments: {tool.arguments}",
                )

            validated_tools.append(tool.simple_output_dump())

        return validated_tools

    def _build_stream_kwargs(
        self,
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        log_params: bool = True,
        **kwargs,
    ) -> dict:
        """Construct provider-specific parameters for streaming API requests."""
        raise NotImplementedError

    async def _stream_chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        stream_kwargs: Optional[dict] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Internal async generator for streaming raw response chunks."""
        raise NotImplementedError

    def _stream_chat_sync(
        self,
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        stream_kwargs: Optional[dict] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Internal synchronous generator for streaming raw response chunks."""
        raise NotImplementedError

    async def _stream_with_retry(
        self,
        operation_name: str,
        messages: list[Message],
        tools: Optional[list[ToolCall]],
        stream_kwargs: dict,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Execute the async streaming operation with retry logic and error recovery."""
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
        messages: list[Message],
        tools: Optional[list[ToolCall]],
        stream_kwargs: dict,
    ) -> Generator[StreamChunk, None, None]:
        """Execute the synchronous streaming operation with retry logic and error recovery."""
        for i in range(self.max_retries):
            try:
                yield from self._stream_chat_sync(messages=messages, tools=tools, stream_kwargs=stream_kwargs)
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
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Public async interface for streaming chat completions with retries."""
        stream_kwargs = self._build_stream_kwargs(messages, tools, **kwargs)
        async for chunk in self._stream_with_retry("stream chat", messages, tools, stream_kwargs):
            yield chunk

    def stream_chat_sync(
        self,
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """Public synchronous interface for streaming chat completions with retries."""
        stream_kwargs = self._build_stream_kwargs(messages, tools, **kwargs)
        yield from self._stream_with_retry_sync("stream chat sync", messages, tools, stream_kwargs)

    async def _chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        enable_stream_print: bool = False,
        **kwargs,
    ) -> Message:
        """Internal async method to aggregate a full response by consuming the stream."""
        state = {
            "enter_think": False,
            "enter_answer": False,
            "reasoning_content": "",
            "answer_content": "",
            "tool_calls": [],
        }

        stream_kwargs = self._build_stream_kwargs(messages, tools, **kwargs)
        async for stream_chunk in self._stream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
            self._process_stream_chunk(stream_chunk, state, enable_stream_print)

        return self._create_message_from_state(state)

    def _chat_sync(
        self,
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        enable_stream_print: bool = False,
        **kwargs,
    ) -> Message:
        """Internal synchronous method to aggregate a full response by consuming the stream."""
        state = {
            "enter_think": False,
            "enter_answer": False,
            "reasoning_content": "",
            "answer_content": "",
            "tool_calls": [],
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
        """Execute a generic async operation with error handling and retry logic."""
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
        """Execute a generic synchronous operation with error handling and retry logic."""
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
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        enable_stream_print: bool = False,
        callback_fn: Optional[Callable[[Message], Any]] = None,
        default_value: Any = None,
        **kwargs,
    ) -> Message | Any:
        """Perform an async chat completion with integrated retries and error handling."""
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
        messages: list[Message],
        tools: Optional[list[ToolCall]] = None,
        enable_stream_print: bool = False,
        callback_fn: Optional[Callable[[Message], Any]] = None,
        default_value: Any = None,
        **kwargs,
    ) -> Message | Any:
        """Perform a synchronous chat completion with integrated retries and error handling."""
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
        """Release any asynchronous resources or connections held by the client."""

    def close_sync(self):
        """Release any synchronous resources or connections held by the client."""
