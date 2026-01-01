"""Base flow module providing abstract flow execution with caching and operation orchestration."""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod

from loguru import logger

from ..context import C, RuntimeContext
from ..enumeration import ChunkEnum, RegistryEnum
from ..op import BaseOp, SequentialOp, ParallelOp
from ..schema import Response, ToolCall, ToolAttr
from ..utils import camel_to_snake, CacheHandler


class BaseFlow(ABC):
    """Abstract base class for flow execution with caching, streaming, and operation tree management.

    BaseFlow provides a framework for building complex workflows by composing operations
    into executable trees. It supports both synchronous and asynchronous execution modes,
    response caching, streaming outputs, and automatic tool call schema generation.
    """

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
        """Initialize flow configuration and execution state."""
        super().__init__()

        self.name: str = name or camel_to_snake(self.__class__.__name__)
        self.stream: bool = stream
        self.raise_exception: bool = raise_exception
        self.enable_cache: bool = enable_cache
        self.cache_path: str = cache_path
        self.cache_expire_hours: float = cache_expire_hours
        self.flow_params: dict = kwargs

        self._flow_op: BaseOp | None = None
        self._cache: CacheHandler | None = None
        self._flow_printed: bool = False
        self._tool_call: ToolCall | None = None

    def _build_tool_call(self) -> ToolCall | None:
        """Generate the tool call schema definition for this flow."""

    @abstractmethod
    def _build_flow(self) -> BaseOp:
        """Construct the root operation tree for flow execution."""

    def _compute_cache_key(self, params: dict) -> str | None:
        """Generate a SHA256 hash from input parameters for caching."""
        try:
            payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.exception(f"{self.name} cache key serialization failed: {e}")
            return None

    def _maybe_load_cached(self, params: dict) -> Response | None:
        """Retrieve a cached response if caching is enabled and available."""
        if not self.enable_cache or self.stream:
            return None

        if key := self._compute_cache_key(params):
            if cached := self.cache.load(key):
                logger.info(f"Loaded {self.name} response from cache.")
                return Response(**cached)
        return None

    def _maybe_save_cache(self, params: dict, response: Response):
        """Persist the execution response to the cache."""
        if not self.enable_cache or self.stream:
            return

        if key := self._compute_cache_key(params):
            self.cache.save(
                key,
                response.model_dump(exclude_none=True),
                expire_hours=self.cache_expire_hours,
            )

    def _print_operation_tree(self, name: str, op: BaseOp, indent: int):
        """Recursively log the hierarchy of the flow's operation tree."""
        prefix = "  " * indent
        op_type = "sequential" if isinstance(op, SequentialOp) else "parallel" if isinstance(op, ParallelOp) else name
        logger.info(f"{prefix}{op_type} execution")

        for sub_op in op.sub_ops or []:
            self._print_operation_tree(sub_op.name, sub_op, indent + 2)

    @property
    def tool_call(self) -> ToolCall | None:
        """Lazily construct the ToolCall schema describing this flow."""
        if self.flow_op.tool_call:
            return self.flow_op.tool_call

        if self._tool_call is None:
            self._tool_call = self._build_tool_call()
            if self._tool_call:
                self._tool_call.name = self._tool_call.name or self.name
                self._tool_call.output = self._tool_call.output or {
                    f"{self.name}_result": ToolAttr(
                        type="string",
                        description=f"The execution result of the {self.name}",
                    ),
                }
        return self._tool_call

    @property
    def cache(self) -> CacheHandler:
        """Provide access to the internal CacheHandler instance."""
        assert self.enable_cache, "Cache usage requested while disabled."
        if self._cache is None:
            self._cache = CacheHandler(f"{self.cache_path}/{self.name}")
        return self._cache

    @property
    def flow_op(self) -> BaseOp:
        """Lazily build and retrieve the root operation of the flow."""
        if self._flow_op is None:
            self._flow_op = self._build_flow()
        return self._flow_op

    @property
    def async_mode(self) -> bool:
        """Check if the current flow operation tree is asynchronous."""
        return self.flow_op.async_mode

    @staticmethod
    def parse_expression(expression: str) -> BaseOp:
        """Parse a string expression into an executable BaseOp instance."""
        lines = [x.strip() for x in expression.strip().splitlines() if x.strip()]
        if not lines:
            raise ValueError("Expression is empty")

        env: dict = C.registry_dict[RegistryEnum.OP]
        if len(lines) > 1:
            exec("\n".join(lines[:-1]), {"__builtins__": {}}, env)

        result = eval(lines[-1], {"__builtins__": {}}, env)
        if not isinstance(result, BaseOp):
            raise TypeError(f"Expression evaluated to {type(result)}, expected BaseOp")
        return result

    def print_flow(self):
        """Log the visual structure of the flow once."""
        if not self._flow_printed:
            logger.info(f"---------- [Flow Structure] {self.name} ----------")
            self._print_operation_tree(self.name, self.flow_op, 0)
            logger.info("-" * 50)
            self._flow_printed = True

    async def call(self, **kwargs) -> Response | asyncio.Queue:
        """Execute the flow asynchronously with parameter caching."""
        kwargs["stream"] = self.stream
        logger.info(f"{self.name} incoming params: {kwargs}")
        if cached := self._maybe_load_cached(kwargs):
            return cached

        context = RuntimeContext(**kwargs)
        try:
            self.print_flow()
            flow_op: BaseOp = self._build_flow()
            assert self.flow_op.async_mode, "Async call requires an async flow operation."

            await flow_op.call(context=context)
            result = context.stream_queue if self.stream else context.response

            if self.stream:
                await context.add_stream_done()

            self._maybe_save_cache(kwargs, result)
            return result
        except Exception as e:
            logger.exception(f"{self.name} async call failed: {e}")
            if self.raise_exception:
                raise e
            if self.stream:
                await context.add_stream_chunk_and_type(str(e), ChunkEnum.ERROR)
                await context.add_stream_done()
                return context.stream_queue
            context.add_response_error(e)
            return context.response

    def call_sync(self, **kwargs) -> Response:
        """Execute the flow synchronously with parameter caching."""
        logger.info(f"{self.name} incoming sync params: {kwargs}")
        assert not self.stream, "Synchronous call cannot be used in stream mode."
        if cached := self._maybe_load_cached(kwargs):
            return cached

        context = RuntimeContext(**kwargs)
        try:
            self.print_flow()
            flow_op: BaseOp = self._build_flow()
            assert not self.flow_op.async_mode, "Sync call requires a sync flow operation."

            flow_op.call_sync(context=context)
            self._maybe_save_cache(kwargs, context.response)
            return context.response
        except Exception as e:
            logger.exception(f"{self.name} sync call failed: {e}")
            if self.raise_exception:
                raise e
            context.add_response_error(e)
            return context.response
