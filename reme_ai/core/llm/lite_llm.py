"""LiteLLM implementation for flowllm.

This module provides an implementation of BaseLLM using the LiteLLM library,
which enables support for 100+ LLM providers through a unified interface.
It handles streaming responses, tool calling, and reasoning content from
supported models. The implementation supports both synchronous and asynchronous
operations with robust error handling and retry logic.
"""

import asyncio
import json
import os
import time
from typing import List, Dict, Optional, Generator, AsyncGenerator

from loguru import logger

from .base_llm import BaseLLM
from ..context import C
from ..enumeration import ChunkEnum
from ..enumeration import Role
from ..schema import FlowStreamChunk
from ..schema import Message
from ..schema import ToolCall


@C.register_llm("litellm")
class LiteLLM(BaseLLM):
    """
    LiteLLM implementation supporting streaming and tool calls.

    This class implements the BaseLLM interface using LiteLLM, which provides
    support for 100+ LLM providers through a unified interface, including:
    - Streaming responses with different chunk types (thinking, answer, tools)
    - Tool calling with parallel execution
    - Reasoning/thinking content from supported models
    - Robust error handling and retries

    The class follows the BaseLLM interface strictly, implementing all required methods
    with proper type annotations and error handling consistent with the base class.

    The implementation aggregates streaming chunks internally in _chat() and _achat()
    methods, which are called by the base class's chat() and achat() methods that add
    retry logic and error handling. Reasoning content is separated from regular answer
    content and stored in the Message's reasoning_content field.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        custom_llm_provider: str = "openai",
        max_retries: int = 5,
        raise_exception: bool = False,
        **kwargs,
    ):
        """
        Initialize the LiteLLM.

        Args:
            model_name: Name of the LLM model to use
            api_key: API key for authentication (defaults to FLOW_LLM_API_KEY env var)
            base_url: Base URL for custom endpoints (defaults to FLOW_LLM_BASE_URL env var)
            custom_llm_provider: Custom LLM provider name for LiteLLM routing
            max_retries: Maximum number of retries on failure (default: 5)
            raise_exception: Whether to raise exception on final failure (default: False)
            **kwargs: Additional parameters passed to the API (e.g., temperature, top_p, seed)
        """
        super().__init__(model_name=model_name, max_retries=max_retries, raise_exception=raise_exception, **kwargs)
        self.api_key: str = api_key or os.getenv("FLOW_LLM_API_KEY", "")
        self.base_url: str = base_url or os.getenv("FLOW_LLM_BASE_URL", "")
        self.custom_llm_provider: str = custom_llm_provider

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        **kwargs,
    ) -> Generator[FlowStreamChunk, None, None]:
        """
        Stream chat completions from LiteLLM.

        This method handles streaming responses and categorizes chunks into different types:
        - THINK: Reasoning/thinking content from the model
        - ANSWER: Regular response content
        - TOOL: Tool calls that need to be executed
        - USAGE: Token usage statistics
        - ERROR: Error information

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            **kwargs: Additional parameters

        Yields:
            FlowStreamChunk for each streaming piece.
            FlowStreamChunk contains chunk_type and chunk content.
        """
        from litellm import completion

        chat_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model_name,
            "custom_llm_provider": self.custom_llm_provider,
            "messages": [x.simple_dump() for x in messages],
            "stream": True,
            "tools": [x.simple_input_dump() for x in tools] if tools else None,
            **self.kwargs,
            **kwargs,
        }
        log_kwargs: Dict[str, object] = {}
        for k, v in chat_kwargs.items():
            if k in ["messages", "tools"]:
                log_kwargs[k] = len(v) if v is not None else 0
            else:
                log_kwargs[k] = v
        logger.info(f"LiteLLM.stream_chat: {log_kwargs}")

        for i in range(self.max_retries):
            try:
                completion = completion(**chat_kwargs)

                ret_tools: List[ToolCall] = []
                is_answering: bool = False

                for chunk in completion:
                    if not chunk.choices:
                        yield FlowStreamChunk(chunk_type=ChunkEnum.USAGE, chunk=chunk.usage.model_dump())

                    else:
                        delta = chunk.choices[0].delta

                        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                            yield FlowStreamChunk(chunk_type=ChunkEnum.THINK, chunk=delta.reasoning_content)

                        else:
                            if not is_answering:
                                is_answering = True

                            if delta.content is not None:
                                yield FlowStreamChunk(chunk_type=ChunkEnum.ANSWER, chunk=delta.content)

                            if delta.tool_calls is not None:
                                for tool_call in delta.tool_calls:
                                    index = tool_call.index

                                    while len(ret_tools) <= index:
                                        ret_tools.append(ToolCall(index=index))

                                    if tool_call.id:
                                        ret_tools[index].id += tool_call.id

                                    if tool_call.function and tool_call.function.name:
                                        ret_tools[index].name += tool_call.function.name

                                    if tool_call.function and tool_call.function.arguments:
                                        ret_tools[index].arguments += tool_call.function.arguments

                if ret_tools:
                    tool_dict: Dict[str, ToolCall] = {x.name: x for x in tools} if tools else {}
                    for tool in ret_tools:
                        if tool.name not in tool_dict:
                            continue

                        if not tool.check_argument():
                            raise ValueError(f"Tool call {tool.name} argument={tool.arguments} are invalid")

                        yield FlowStreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool.simple_output_dump())

                return

            except Exception as e:
                logger.exception(f"stream chat with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e

                    yield FlowStreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    return

                yield FlowStreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                time.sleep(1 + i)

    async def astream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        **kwargs,
    ) -> AsyncGenerator[FlowStreamChunk, None]:
        """
        Async stream chat completions from LiteLLM.

        This method handles async streaming responses and categorizes chunks into different types:
        - THINK: Reasoning/thinking content from the model
        - ANSWER: Regular response content
        - TOOL: Tool calls that need to be executed
        - USAGE: Token usage statistics
        - ERROR: Error information

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            **kwargs: Additional parameters

        Yields:
            FlowStreamChunk for each streaming piece.
            FlowStreamChunk contains chunk_type, chunk content, and metadata.
        """
        from litellm import acompletion

        chat_kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model_name,
            "custom_llm_provider": self.custom_llm_provider,
            "messages": [x.simple_dump() for x in messages],
            "stream": True,
            "tools": [x.simple_input_dump() for x in tools] if tools else None,
            **self.kwargs,
            **kwargs,
        }
        log_kwargs: Dict[str, object] = {}
        for k, v in chat_kwargs.items():
            if k in ["messages", "tools"]:
                log_kwargs[k] = len(v) if v is not None else 0
            else:
                log_kwargs[k] = v
        logger.info(f"LiteLLM.astream_chat: {log_kwargs}")

        for i in range(self.max_retries):
            try:
                completion = await acompletion(**chat_kwargs)

                ret_tools: List[ToolCall] = []
                is_answering: bool = False

                async for chunk in completion:
                    if not chunk.choices:
                        yield FlowStreamChunk(chunk_type=ChunkEnum.USAGE, chunk=chunk.usage.model_dump())

                    else:
                        delta = chunk.choices[0].delta

                        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                            yield FlowStreamChunk(chunk_type=ChunkEnum.THINK, chunk=delta.reasoning_content)

                        else:
                            if not is_answering:
                                is_answering = True

                            if delta.content is not None:
                                yield FlowStreamChunk(chunk_type=ChunkEnum.ANSWER, chunk=delta.content)

                            if delta.tool_calls is not None:
                                for tool_call in delta.tool_calls:
                                    index = tool_call.index

                                    while len(ret_tools) <= index:
                                        ret_tools.append(ToolCall(index=index))

                                    if tool_call.id:
                                        ret_tools[index].id += tool_call.id

                                    if tool_call.function and tool_call.function.name:
                                        ret_tools[index].name += tool_call.function.name

                                    if tool_call.function and tool_call.function.arguments:
                                        ret_tools[index].arguments += tool_call.function.arguments

                if ret_tools:
                    tool_dict: Dict[str, ToolCall] = {x.name: x for x in tools} if tools else {}
                    for tool in ret_tools:
                        if tool.name not in tool_dict:
                            continue

                        if not tool.check_argument():
                            raise ValueError(f"Tool call {tool.name} argument={tool.arguments} are invalid")

                        yield FlowStreamChunk(chunk_type=ChunkEnum.TOOL, chunk=tool.simple_output_dump())

                return

            except Exception as e:
                logger.exception(f"stream chat with model={self.model_name} encounter error with e={e.args}")

                if i == self.max_retries - 1:
                    if self.raise_exception:
                        raise e

                    yield FlowStreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                    return

                yield FlowStreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e))
                await asyncio.sleep(1 + i)

    def _chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        enable_stream_print: bool = False,
        **kwargs,
    ) -> Message:
        """
        Internal method to perform a single chat completion by aggregating streaming chunks.

        This method is called by the base class's chat() method which adds retry logic
        and error handling. It consumes the entire streaming response from stream_chat()
        and combines all chunks into a single Message object. It separates reasoning content,
        regular answer content, and tool calls, providing a complete response.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            enable_stream_print: Whether to print streaming response to console
            **kwargs: Additional parameters

        Returns:
            Complete Message with all content aggregated from streaming chunks
        """

        enter_think = False
        enter_answer = False
        reasoning_content = ""
        answer_content = ""
        tool_calls = []

        for stream_chunk in self.stream_chat(messages, tools, **kwargs):
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

    async def _achat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolCall]] = None,
        enable_stream_print: bool = False,
        **kwargs,
    ) -> Message:
        """
        Internal async method to perform a single chat completion by aggregating streaming chunks.

        This method is called by the base class's achat() method which adds retry logic
        and error handling. It consumes the entire async streaming response from astream_chat()
        and combines all chunks into a single Message object. It separates reasoning content,
        regular answer content, and tool calls, providing a complete response.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools available to the model
            enable_stream_print: Whether to print streaming response to console
            **kwargs: Additional parameters

        Returns:
            Complete Message with all content aggregated from streaming chunks
        """

        enter_think = False
        enter_answer = False
        reasoning_content = ""
        answer_content = ""
        tool_calls = []

        async for stream_chunk in self.astream_chat(messages, tools, **kwargs):
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
