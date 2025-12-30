import asyncio
import hashlib
import json
from abc import ABC
from functools import partial
from typing import Optional

from loguru import logger

from ..context import RuntimeContext, C
from ..enumeration import ChunkEnum
from ..op import BaseOp, SequentialOp, ParallelOp
from ..schema import Response, StreamChunk, ToolCall, ToolAttr
from ..utils import camel_to_snake, CacheHandler


class BaseFlow(ABC):

    def __init__(
        self,
        name: str = "",
        stream: bool = False,
        raise_exception: bool = True,
        enable_cache: bool = False,
        cache_path: str = "cache/flow",
        cache_expire_hours: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.name: str = name or camel_to_snake(self.__class__.__name__)
        self.stream: bool = stream
        self.raise_exception: bool = raise_exception
        self.enable_cache: bool = enable_cache
        self.cache_path: str = cache_path
        self.cache_expire_hours: float = cache_expire_hours
        self.flow_params: dict = kwargs

        self._flow_op: Optional[BaseOp] = None
        self._cache: CacheHandler | None = None
        self._flow_printed: bool = False
        self._tool_call: ToolCall | None = None

    def _build_tool_call(self) -> ToolCall | None:
        """Build and return the tool call schema for this flow.

        Example:
            ```python
            def build_tool_call(self) -> ToolCall:
                return ToolCall(
                    **{
                        "description": "Search the web for information",
                        "input_schema": {
                            "query": {
                                "type": "string",
                                "description":"Search query",
                                "required": True,
                            },
                        },
                    }
                )
            ```
        """

    @property
    def tool_call(self) -> ToolCall | None:
        """Return the lazily constructed `ToolCall` describing this tool."""
        if self._tool_call is None:
            self._tool_call = self._build_tool_call()
            if self._tool_call is None:
                return None

            if not self._tool_call.name:
                self._tool_call.name = self.name

            if not self._tool_call.output_schema:
                self._tool_call.output_schema = {
                    f"{self.name}_result": ToolAttr(
                        type="string",
                        description=f"The execution result of the {self.name}",
                    ),
                }

        return self._tool_call

    @property
    def cache(self):
        assert self.enable_cache, "cache is disabled!"
        if self._cache is None:
            self._cache = CacheHandler(f"{self.cache_path}/{self.name}")
        return self._cache

    def _compute_cache_key(self, params: dict) -> Optional[str]:
        try:
            payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.exception(f"{self.name} cache key serialization failed: {e}")
            return None

    def _maybe_load_cached(self, params: dict) -> Optional[Response]:
        if not self.enable_cache or self.stream:
            return None

        key = self._compute_cache_key(params)
        if not key:
            return None

        cached = self.cache.load(key)
        if cached is not None:
            logger.info(f"load flow response from cache with params={params}")
            return Response(**cached)

        return None

    def _maybe_save_cache(self, params: dict, response: Response):
        if not self.enable_cache or self.stream:
            return

        key = self._compute_cache_key(params)
        if not key:
            return

        self.cache.save(key, response.model_dump(), expire_hours=self.cache_expire_hours)

    @property
    def async_mode(self) -> bool:
        return self.flow_op.async_mode

    def _build_flow(self) -> BaseOp:
        """Build and return the root `BaseOp` for this flow."""

    @property
    def flow_op(self) -> BaseOp:
        """Lazily build and cache the root operation for this flow."""
        if self._flow_op is None:
            self._flow_op = self._build_flow()
        return self._flow_op

    def print_flow(self):
        if not self._flow_printed:
            logger.info(f"---------- [Flow Structure] {self.name} ----------")
            self._print_operation_tree(self.name, self.flow_op, indent=0)
            logger.info(f"--------------------------------------------------")
            self._flow_printed = True

    def _print_operation_tree(self, name: str, op: BaseOp, indent: int):
        prefix = "  " * indent
        op_name = "sequential" if isinstance(op, SequentialOp) else "parallel" if isinstance(op, ParallelOp) else name
        logger.info(f"{prefix}{op_name} execution")

        if op.sub_ops:
            for sub_name, sub_op in op.sub_ops.items():
                self._print_operation_tree(sub_name, sub_op, indent + 2)

    async def call(self, **kwargs) -> Response | StreamChunk | None:
        kwargs["stream"] = self.stream
        logger.info(f"{self.name} incoming params: {kwargs}")

        cached = self._maybe_load_cached(kwargs)
        if cached is not None:
            return cached

        context = RuntimeContext(**kwargs)
        try:
            self.print_flow()
            flow_op: BaseOp = self._build_flow()

            if self.async_mode:
                await flow_op.call(context=context)

            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(C.thread_pool, partial(flow_op.call_sync, context=context))

            if self.stream:
                await context.add_stream_done()
                result = context.stream_queue
            else:
                result = context.response

            self._maybe_save_cache(kwargs, result)
            return result

        except Exception as e:
            logger.exception(f"{self.name} async call encounter error={e.args}")
            if self.raise_exception:
                raise e

            if self.stream:
                await context.add_stream_chunk_and_type(str(e), ChunkEnum.ERROR)
                await context.add_stream_done()
                return context.stream_queue

            else:
                context.add_response_error(e)
                return context.response

    def call_sync(self, **kwargs) -> Response:
        kwargs["stream"] = self.stream
        logger.info(f"{self.name} incoming params (sync): {kwargs}")

        cached = self._maybe_load_cached(kwargs)
        if cached is not None:
            return cached

        context = RuntimeContext(**kwargs)

        try:
            self.print_flow()
            flow_op: BaseOp = self._build_flow()

            if self.async_mode:
                asyncio.run(flow_op.call(context=context))
            else:
                flow_op.call_sync(context=context)
            result = context.response

            self._maybe_save_cache(kwargs, result)
            return result

        except Exception as e:
            logger.exception(f"{self.name} call encounter error={e.args}")
            if self.raise_exception:
                raise e

            context.add_response_error(e)
            return context.response
