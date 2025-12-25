import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Callable, Generator, AsyncGenerator, Any

from loguru import logger

from ..schema import Message
from ..schema import StreamChunk
from ..schema import ToolCall


class BaseLLM(ABC):
    def __init__(self, model_name: str, max_retries: int = 5, raise_exception: bool = False, **kwargs):
        self.model_name: str = model_name
        self.max_retries: int = max_retries
        self.raise_exception: bool = raise_exception
        self.kwargs: dict = kwargs  # qwen: https://help.aliyun.com/zh/model-studio/qwen-api-reference

    @abstractmethod
    def stream_chat(
            self,
            messages: List[Message],
            tools: List[ToolCall] | None = None,
            **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """Stream chat completions from the LLM."""

    @abstractmethod
    async def astream_chat(
            self,
            messages: List[Message],
            tools: List[ToolCall] | None = None,
            **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async stream chat completions from the LLM."""

    @abstractmethod
    def _chat(
            self,
            messages: List[Message],
            tools: List[ToolCall] | None = None,
            enable_stream_print: bool = False,
            **kwargs,
    ) -> Message:
        """Internal method to perform a single chat completion."""

    @abstractmethod
    async def _achat(
            self,
            messages: List[Message],
            tools: List[ToolCall] | None = None,
            enable_stream_print: bool = False,
            **kwargs,
    ) -> Message:
        """Internal async method to perform a single chat completion."""

    def chat(
            self,
            messages: List[Message],
            tools: List[ToolCall] | None = None,
            enable_stream_print: bool = False,
            callback_fn: Callable[[Message], Any] | None = None,
            default_value: Any = None,
            **kwargs,
    ) -> Message | Any:
        """Perform a chat completion with retry logic and error handling."""

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

        return default_value

    async def achat(
            self,
            messages: List[Message],
            tools: List[ToolCall] | None = None,
            enable_stream_print: bool = False,
            callback_fn: Callable[[Message], Any] | None = None,
            default_value: Any = None,
            **kwargs,
    ) -> Message | Any:
        """Perform an async chat completion with retry logic and error handling."""

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

        return default_value

    def close(self):
        """Close the client connection or clean up resources."""

    async def async_close(self):
        """Asynchronously close the client connection or clean up resources."""
