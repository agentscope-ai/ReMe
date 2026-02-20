"""Base interface for LLM implementations."""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Generator, AsyncGenerator, Any

from loguru import logger

from ..enumeration import ChunkEnum, Role
from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall
from ..utils import extract_content


class BaseLLM(ABC):
    """Base class for LLM interactions."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = "",
        max_retries: int = 10,
        raise_exception: bool = False,
        request_interval: float = 0.0,
        **kwargs,
    ):
        """Initialize LLM client.

        Args:
            model_name: Model name to use
            max_retries: Maximum retry attempts on failure
            raise_exception: Raise exceptions or return default values
            request_interval: Minimum seconds between requests (default: 0.0)
            **kwargs: Additional model-specific parameters
        """
        self._api_key: str = api_key
        self._base_url: str = base_url
        self.model_name: str = model_name
        self.max_retries: int = max_retries
        self.raise_exception: bool = raise_exception
        self.request_interval: float = request_interval
        self.kwargs: dict = kwargs

        self._last_request_time: float = 0.0
        self._request_lock: asyncio.Lock = asyncio.Lock()

    @property
    def api_key(self) -> str | None:
        """Get API key from environment variable."""
        return os.getenv("REME_LLM_API_KEY") or self._api_key

    @property
    def base_url(self) -> str | None:
        """Get base URL from environment variable."""
        return os.getenv("REME_LLM_BASE_URL") or self._base_url

    @staticmethod
    def _accumulate_tool_call_chunk(tool_call, ret_tools: list[ToolCall]):
        """Assemble incremental tool call chunks into complete ToolCall objects."""
        index = tool_call.index

        while len(ret_tools) <= index:
            ret_tools.append(ToolCall(index=index))

        if tool_call.id:
            ret_tools[index].id += tool_call.id

        if tool_call.function and tool_call.function.name:
            ret_tools[index].name += tool_call.function.name

        if tool_call.function and tool_call.function.arguments:
            ret_tools[index].arguments += tool_call.function.arguments

    @staticmethod
    def _validate_and_serialize_tools(ret_tool_calls: list[ToolCall], tools: list[ToolCall]) -> list[dict]:
        """Validate and serialize tool calls."""
        if not ret_tool_calls:
            return []

        tool_dict: dict[str, ToolCall] = {x.name: x for x in tools} if tools else {}
        validated_tools = []

        for tool in ret_tool_calls:
            if tool.name not in tool_dict:
                continue

            if not tool.sanitize_and_check_argument():
                logger.error(f"Invalid JSON arguments in {tool.name}: {tool.arguments}")
                raise ValueError(f"Invalid JSON arguments in {tool.name}: {tool.arguments}")

            validated_tools.append(tool.simple_output_dump())
        return validated_tools

    @abstractmethod
    def _build_stream_kwargs(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        log_params: bool = True,
        model_name: str | None = None,
        **kwargs,
    ) -> dict:
        """Build provider-specific streaming parameters."""

    async def _stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None,
        stream_kwargs: dict,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async generator for streaming response chunks."""

    def _stream_chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        stream_kwargs: dict | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Sync generator for streaming response chunks."""

    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat completions with retries and return final message."""
        if self.request_interval > 0:
            async with self._request_lock:
                current_time = time.time()
                elapsed = current_time - self._last_request_time
                if elapsed < self.request_interval:
                    await asyncio.sleep(self.request_interval - elapsed)
                self._last_request_time = time.time()

        async for chunk in self._stream_chat_impl(messages, tools, model_name, **kwargs):
            yield chunk

    async def _stream_chat_impl(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat with retry logic."""
        stream_kwargs = self._build_stream_kwargs(messages, tools, model_name=model_name, **kwargs)

        for i in range(self.max_retries):
            try:
                async for chunk in self._stream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
                    yield chunk

                break

            except Exception as e:
                logger.exception(f"Stream chat error (model={self.model_name}): {e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    break

                yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                await asyncio.sleep(i + 1)

    def stream_chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        model_name: str | None = None,
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """Stream chat completions synchronously with retries."""
        stream_kwargs = self._build_stream_kwargs(messages, tools, model_name=model_name, **kwargs)

        for i in range(self.max_retries):
            try:
                yield from self._stream_chat_sync(messages=messages, tools=tools, stream_kwargs=stream_kwargs)
                break

            except Exception as e:
                logger.exception(f"Stream chat sync error (model={self.model_name}): {e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    break

                yield StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                time.sleep(i + 1)

    async def _chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        model_name: str | None = None,
        **kwargs,
    ) -> Message:
        """Aggregate full response by consuming the stream."""
        state = {
            "enter_think": False,
            "enter_answer": False,
            "reasoning_content": "",
            "answer_content": "",
            "tool_calls": [],
        }

        stream_kwargs = self._build_stream_kwargs(messages, tools, model_name=model_name, **kwargs)
        async for stream_chunk in self._stream_chat(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
            if stream_chunk.chunk_type is ChunkEnum.USAGE:
                if enable_stream_print:
                    print(
                        f"\n<usage>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</usage>",
                        flush=True,
                    )

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

        return Message(
            role=Role.ASSISTANT,
            reasoning_content=state["reasoning_content"],
            content=state["answer_content"],
            tool_calls=state["tool_calls"],
        )

    def _chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        model_name: str | None = None,
        **kwargs,
    ) -> Message:
        """Aggregate full response synchronously by consuming the stream."""
        state = {
            "enter_think": False,
            "enter_answer": False,
            "reasoning_content": "",
            "answer_content": "",
            "tool_calls": [],
        }

        stream_kwargs = self._build_stream_kwargs(messages, tools, model_name=model_name, **kwargs)
        for stream_chunk in self._stream_chat_sync(messages=messages, tools=tools, stream_kwargs=stream_kwargs):
            if stream_chunk.chunk_type is ChunkEnum.USAGE:
                if enable_stream_print:
                    print(
                        f"\n<usage>{json.dumps(stream_chunk.chunk, ensure_ascii=False, indent=2)}</usage>",
                        flush=True,
                    )

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

        return Message(
            role=Role.ASSISTANT,
            reasoning_content=state["reasoning_content"],
            content=state["answer_content"],
            tool_calls=state["tool_calls"],
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        callback_fn: Callable[[Message], Any] | None = None,
        default_value: Any = None,
        model_name: str | None = None,
        **kwargs,
    ) -> Message | Any:
        """Chat completion with retries and error handling."""
        if self.request_interval > 0:
            async with self._request_lock:
                current_time = time.time()
                elapsed = current_time - self._last_request_time
                if elapsed < self.request_interval:
                    await asyncio.sleep(self.request_interval - elapsed)
                self._last_request_time = time.time()

        return await self._chat_impl(
            messages,
            tools,
            enable_stream_print,
            callback_fn,
            default_value,
            model_name,
            **kwargs,
        )

    async def _chat_impl(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        callback_fn: Callable[[Message], Any] | None = None,
        default_value: Any = None,
        model_name: str | None = None,
        **kwargs,
    ) -> Message | Any:
        """Chat with retry and error handling logic."""
        effective_model = model_name if model_name is not None else self.model_name

        for i in range(self.max_retries):
            try:
                result = await self._chat(
                    messages=messages,
                    tools=tools,
                    enable_stream_print=enable_stream_print,
                    model_name=model_name,
                    **kwargs,
                )
                return callback_fn(result) if callback_fn else result

            except Exception as e:
                error_message = str(e.args[0]) if e.args else str(e)
                is_inappropriate_content = "inappropriate content" in error_message.lower()
                is_rate_limit_error = (
                    "request rate increased too quickly" in error_message.lower()
                    or "exceeded your current quota" in error_message.lower()
                    or "insufficient_quota" in error_message.lower()
                )

                if is_inappropriate_content:
                    logger.error(f"Inappropriate content detected (model={effective_model})")
                    logger.error("=" * 80)
                    for idx, msg in enumerate(messages):
                        logger.error(f"Message {idx + 1} [role={msg.role}]:")
                        logger.error(f"Content: {msg.content}")
                        if msg.reasoning_content:
                            logger.error(f"Reasoning: {msg.reasoning_content}")
                        if msg.tool_calls:
                            logger.error(f"Tool calls: {msg.tool_calls}")
                        logger.error("-" * 80)
                    logger.error("=" * 80)
                    return Message(role=Role.ASSISTANT, content="")

                if is_rate_limit_error:
                    logger.warning(
                        f"Rate limit hit (model={effective_model}), sleeping 60s (attempt {i + 1}/{self.max_retries})",
                    )
                    await asyncio.sleep(60)
                    continue

                logger.exception(f"Chat error (model={effective_model}): {e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    return default_value

                await asyncio.sleep(1 + i)
        return default_value

    def chat_sync(
        self,
        messages: list[Message],
        tools: list[ToolCall] | None = None,
        enable_stream_print: bool = False,
        callback_fn: Callable[[Message], Any] | None = None,
        default_value: Any = None,
        model_name: str | None = None,
        **kwargs,
    ) -> Message | Any:
        """Chat completion synchronously with retries and error handling."""
        effective_model = model_name if model_name is not None else self.model_name

        for i in range(self.max_retries):
            try:
                result = self._chat_sync(
                    messages=messages,
                    tools=tools,
                    enable_stream_print=enable_stream_print,
                    model_name=model_name,
                    **kwargs,
                )
                return callback_fn(result) if callback_fn else result

            except Exception as e:
                error_message = str(e.args[0]) if e.args else str(e)
                is_inappropriate_content = "inappropriate content" in error_message.lower()
                is_rate_limit_error = (
                    "request rate increased too quickly" in error_message.lower()
                    or "exceeded your current quota" in error_message.lower()
                    or "insufficient_quota" in error_message.lower()
                )

                if is_inappropriate_content:
                    logger.error(f"Inappropriate content detected (model={effective_model})")
                    logger.error("=" * 80)
                    for idx, msg in enumerate(messages):
                        logger.error(f"Message {idx + 1} [role={msg.role}]:")
                        logger.error(f"Content: {msg.content}")
                        if msg.reasoning_content:
                            logger.error(f"Reasoning: {msg.reasoning_content}")
                        if msg.tool_calls:
                            logger.error(f"Tool calls: {msg.tool_calls}")
                        logger.error("-" * 80)
                    logger.error("=" * 80)
                    return Message(role=Role.ASSISTANT, content="")

                if is_rate_limit_error:
                    logger.warning(
                        f"Rate limit hit (model={effective_model}), sleeping 60s (attempt {i + 1}/{self.max_retries})",
                    )
                    time.sleep(60)
                    continue

                logger.exception(f"Chat sync error (model={effective_model}): {e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e
                    return default_value

                time.sleep(1 + i)
        return default_value

    async def simple_request(
        self,
        prompt: str,
        model_name: str | None = None,
        callback_fn: Callable[[Message], Any] | None = None,
        default_value: Any = None,
        **kwargs,
    ) -> str:
        """Make a simple request using the LLM."""
        assistant_message = await self.chat(
            messages=[Message(role=Role.USER, content=prompt)],
            model_name=model_name,
            callback_fn=callback_fn,
            default_value=default_value,
            **kwargs,
        )
        return assistant_message.content

    async def simple_request_for_json(
        self,
        prompt: str,
        model_name: str | None = None,
        **kwargs,
    ) -> dict:
        """Make a simple request using the LLM and extract JSON."""

        def extract_fn(message: Message) -> dict:
            return extract_content(message.content)

        return await self.chat(
            messages=[Message(role=Role.USER, content=prompt)],
            model_name=model_name,
            callback_fn=extract_fn,
            default_value={},
            **kwargs,
        )

    async def close(self):
        """Release async resources."""

    def close_sync(self):
        """Release sync resources."""
