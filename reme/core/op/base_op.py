"""Base operator class for LLM workflow execution and composition."""

import asyncio
import copy
import inspect
from abc import ABCMeta
from pathlib import Path
from typing import Callable, Optional, Any

from loguru import logger
from tqdm import tqdm

from ..embedding import BaseEmbeddingModel
from ..file_store import BaseFileStore
from ..llm import BaseLLM
from ..prompt_handler import PromptHandler
from ..runtime_context import RuntimeContext
from ..schema import Response, ServiceConfig
from ..schema.service_config import OpConfig
from ..service_context import ServiceContext
from ..token_counter import BaseTokenCounter
from ..utils import camel_to_snake, CacheHandler, timer
from ..vector_store import BaseVectorStore


class BaseOp(metaclass=ABCMeta):
    """Base operator class for LLM workflow execution and composition."""

    __alias_name__: str = ""

    def __new__(cls, *args, **kwargs):
        """Capture initialization arguments for object cloning."""
        instance = super().__new__(cls)
        instance._init_args = copy.copy(args)
        instance._init_kwargs = copy.copy(kwargs)
        return instance

    def __init__(
        self,
        name: str = "",
        async_mode: bool = True,
        language: str = "",
        prompt_name: str = "",
        prompt_path: str = "",
        llm: str | BaseLLM = "default",
        embedding_model: str | BaseEmbeddingModel = "default",
        vector_store: str | BaseVectorStore = "default",
        file_store: str | BaseFileStore = "default",
        token_counter: str | BaseTokenCounter = "default",
        enable_cache: bool = False,
        cache_path: str = "cache/op",
        cache_expire_hours: float | None = None,
        sub_ops: dict[str, "BaseOp"] | list["BaseOp"] | Optional["BaseOp"] = None,
        input_mapping: dict[str, str] | None = None,
        output_mapping: dict[str, str] | None = None,
        enable_parallel: bool = False,
        max_retries: int = 1,
        raise_exception: bool = False,
        **kwargs,
    ):
        """Initialize operator configurations and internal state."""
        self.name = name or self.__alias_name__ or camel_to_snake(self.__class__.__name__)
        self.async_mode = async_mode
        self.language = language
        self.prompt = self._get_prompt_handler(prompt_name, prompt_path)

        self._llm = llm
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._file_store = file_store
        self._token_counter = token_counter

        self.enable_cache = enable_cache
        self.cache_path = cache_path
        self.cache_expire_hours = cache_expire_hours

        self.sub_ops: list["BaseOp"] = []
        self.add_sub_ops(sub_ops)

        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.enable_parallel = enable_parallel  # Control whether to execute tasks in parallel
        self.max_retries = max(1, max_retries)
        self.raise_exception = raise_exception
        self.op_params = kwargs

        self._pending_tasks: list = []
        self.context: RuntimeContext | None = None
        self._cache: CacheHandler | None = None

    def _get_prompt_handler(self, prompt_name: str, prompt_path: str) -> PromptHandler:
        """Load prompt configuration from the associated YAML file."""
        if prompt_path:
            path = Path(prompt_path)
        else:
            path = Path(inspect.getfile(self.__class__))
            if prompt_name:
                path = path.with_stem(prompt_name)
        return PromptHandler(language=self.language).load_prompt_by_file(path.with_suffix(".yaml"))

    def _handle_failure(self, e: Exception, attempt: int) -> str | None:
        """Log failures and handle final retry logic."""
        message = f"[{self.__class__.__name__}] failed (attempt {attempt + 1}): {e}"
        if attempt == self.max_retries - 1:
            logger.exception(message)
            if self.raise_exception:
                raise e
            return f"[{self.__class__.__name__}] failed: {e}"
        else:
            logger.warning(message)
            return None

    @property
    def cache(self) -> CacheHandler:
        """Access the operator-specific cache handler."""
        assert self.enable_cache, "Cache is disabled!"
        if not self._cache:
            self._cache = CacheHandler(f"{self.cache_path}/{self.name}")
        return self._cache

    @property
    def service_context(self) -> ServiceContext:
        """Access the service context."""
        assert self.context, "Service context is not initialized!"
        return self.context.service_context

    @property
    def service_config(self) -> ServiceConfig:
        """Access the service configuration."""
        return self.service_context.service_config

    @property
    def llm(self) -> BaseLLM:
        """Get the LLM instance from ServiceContext."""
        if isinstance(self._llm, str):
            self._llm = self.service_context.llms[self._llm]
        return self._llm

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        """Get the embedding model instance from ServiceContext."""
        if isinstance(self._embedding_model, str):
            self._embedding_model = self.service_context.embedding_models[self._embedding_model]
        return self._embedding_model

    @property
    def vector_store(self) -> BaseVectorStore:
        """Lazily initialize and return the vector store instance."""
        if isinstance(self._vector_store, str):
            self._vector_store = self.service_context.vector_stores[self._vector_store]
        return self._vector_store

    @property
    def file_store(self) -> BaseFileStore:
        """Lazily initialize and return the file store instance."""
        if isinstance(self._file_store, str):
            self._file_store = self.service_context.file_stores[self._file_store]
        return self._file_store

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Get the token counter instance from ServiceContext."""
        if isinstance(self._token_counter, str):
            self._token_counter = self.service_context.token_counters[self._token_counter]
        return self._token_counter

    @property
    def service_metadata(self) -> dict:
        """Get service configuration metadata."""
        return self.service_context.service_config.metadata

    @property
    def response(self) -> Response:
        """Access the response object."""
        return self.context.response

    def before_execute_sync(self):
        """Prepare context and validate before sync execution.

        This method performs the following steps:
        1. Apply input mapping to transform context variables
        2. Load operator-specific configuration from service config if available
        3. Override operator parameters and prompts based on config
        """
        self.context.apply_mapping(self.input_mapping)

        if self.context.service_context is None:
            return

        service_config = self.service_context.service_config
        if self.name not in service_config.ops:
            return

        op_config: OpConfig = service_config.ops[self.name]

        # Override operator parameters from config
        if op_config.params:
            for k, v in op_config.params.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                    logger.info(f"[{self.__class__.__name__}] Set attribute '{k}' = {v}")
                else:
                    self.op_params[k] = v
                    logger.info(f"[{self.__class__.__name__}] Set op_param '{k}' = {v}")

        # Load custom prompt templates from config
        if op_config.prompt_dict:
            self.prompt.load_prompt_dict(op_config.prompt_dict)
            logger.info(f"[{self.__class__.__name__}] Loaded prompt keys={list(op_config.prompt_dict.keys())}")

    async def before_execute(self):
        """Prepare context and validate before async execution."""
        self.before_execute_sync()

    def execute_sync(self):
        """Define core sync logic in subclasses."""

    async def execute(self):
        """Define core async logic in subclasses."""

    def after_execute_sync(self, response: Any):
        """Finalize context and mappings after sync execution."""
        self.context.apply_mapping(self.output_mapping)
        if response is not None:
            if isinstance(response, dict):
                for k, v in response.items():
                    if k == "answer":
                        self.response.answer = v
                    elif k == "success":
                        self.response.success = v if isinstance(v, bool) else v.lower() == "true"
                    else:
                        self.response.metadata[k] = v
            else:
                self.response.answer = response
        return response

    async def after_execute(self, output: Any):
        """Finalize context and mappings after async execution."""
        return self.after_execute_sync(output)

    @timer
    def call_sync(self, context: RuntimeContext = None, **kwargs):
        """Execute the operator synchronously with retry logic."""
        self.context = RuntimeContext.from_context(context, **kwargs)
        response = None
        for i in range(self.max_retries):
            try:
                self.before_execute_sync()
                response = self.execute_sync()
                response = self.after_execute_sync(response)
                break
            except Exception as e:
                response = self._handle_failure(e, i)

        return response

    @timer
    async def call(self, context: RuntimeContext = None, **kwargs):
        """Execute the operator asynchronously with retry logic."""
        self.context = RuntimeContext.from_context(context, **kwargs)
        response = None
        for i in range(self.max_retries):
            try:
                await self.before_execute()
                response = await self.execute()
                response = await self.after_execute(response)
                break
            except Exception as e:
                response = self._handle_failure(e, i)
        return response

    def submit_sync_task(self, fn: Callable, *args, **kwargs) -> "BaseOp":
        """Submit a task to the thread pool or local queue."""
        if self.enable_parallel:
            task = self.service_context.thread_pool.submit(fn, *args, **kwargs)
        else:
            task = (fn, args, kwargs)
        self._pending_tasks.append(task)
        return self

    def submit_async_task(self, coro_fn: Callable, *args, **kwargs) -> "BaseOp":
        """Submit an async task to the pending tasks queue."""
        task = coro_fn(*args, **kwargs)
        self._pending_tasks.append(task)
        return self

    def join_sync_tasks(self, task_desc: str = None) -> list:
        """Wait for all pending sync tasks and return flattened results."""
        results = []
        for task in tqdm(self._pending_tasks, desc=task_desc or self.name):
            if self.enable_parallel:
                result = task.result()
            else:
                result = task[0](*task[1], **task[2])
            if result:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
        self._pending_tasks.clear()
        return results

    async def join_async_tasks(self, return_exceptions: bool = True) -> list:
        """Wait for all pending async tasks and aggregate results."""
        if self.enable_parallel:
            raw_results = await asyncio.gather(*self._pending_tasks, return_exceptions=return_exceptions)
        else:
            raw_results = []
            for task in self._pending_tasks:
                try:
                    result = await task
                    raw_results.append(result)
                except Exception as e:
                    if return_exceptions:
                        raw_results.append(e)
                    else:
                        raise

        results = []
        for result in raw_results:
            if isinstance(result, Exception):
                logger.error(f"[{self.__class__.__name__}] Async task failed: {result}")
            elif result:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
        self._pending_tasks.clear()
        return results

    def add_sub_ops(self, sub_ops: dict[str, "BaseOp"] | list["BaseOp"] | Optional["BaseOp"]):
        """Add child operators to this operator's sub_ops."""
        if not sub_ops:
            return

        if isinstance(sub_ops, dict):
            for name, op in sub_ops.items():
                assert self.async_mode == op.async_mode, "Async mode mismatch!"
                op.name = name
                if self.language:
                    op.language = self.language
                self.sub_ops.append(op)

        elif isinstance(sub_ops, list):
            for op in sub_ops:
                assert self.async_mode == op.async_mode, "Async mode mismatch!"
                if self.language:
                    op.language = self.language
                self.sub_ops.append(op)

        else:
            assert self.async_mode == sub_ops.async_mode, "Async mode mismatch!"
            if self.language:
                sub_ops.language = self.language
            self.sub_ops.append(sub_ops)

    def add_sub_op(self, sub_op: "BaseOp"):
        """Add a single child operator to this operator's sub_ops."""
        self.sub_ops.append(sub_op)

    def __lshift__(self, ops):
        """Operator overload for adding sub-operators."""
        self.add_sub_ops(ops)
        return self

    def __rshift__(self, op: "BaseOp"):
        """Operator overload for sequential execution composition."""
        from .sequential_op import SequentialOp

        seq = SequentialOp(sub_ops=[self], async_mode=self.async_mode)
        seq.add_sub_ops(op.sub_ops if isinstance(op, SequentialOp) else op)
        return seq

    def __or__(self, op: "BaseOp"):
        """Operator overload for parallel execution composition."""
        from .parallel_op import ParallelOp

        par = ParallelOp(sub_ops=[self], async_mode=self.async_mode)
        par.add_sub_ops(op.sub_ops if isinstance(op, ParallelOp) else op)
        return par

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        """Format a prompt template with provided keyword arguments."""
        return self.prompt.prompt_format(prompt_name=prompt_name, **kwargs)

    def get_prompt(self, prompt_name: str) -> str:
        """Get a prompt template by name."""
        return self.prompt.get_prompt(prompt_name=prompt_name)

    def copy(self, **kwargs):
        """Create a copy of this operator with optional parameter overrides."""
        copy_op = self.__class__(*self._init_args, **{**self._init_kwargs, **kwargs})
        if self.sub_ops:
            copy_op.sub_ops.clear()
            for op in self.sub_ops:
                copy_op.add_sub_op(op.copy())
        return copy_op
