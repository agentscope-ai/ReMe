import asyncio
import copy
import inspect
from pathlib import Path
from typing import Callable, List, Any, Dict, Union

from loguru import logger
from tqdm import tqdm

from ..context import RuntimeContext, PromptHandler, C, BaseContext
from ..embedding import BaseEmbeddingModel
from ..llm import BaseLLM
from ..schema import LLMConfig, EmbeddingModelConfig, TokenCounterConfig, VectorStoreConfig, ToolCall, ToolAttr
from ..token_counter import BaseTokenCounter
from ..utils import Timer, camel_to_snake, CacheHandler
from ..vector_store import BaseVectorStore


class BaseOp:

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._init_args = copy.copy(args)
        instance._init_kwargs = copy.copy(kwargs)
        return instance

    def __init__(
        self,
        name: str = "",
        async_mode: bool = True,
        language: str = "",
        prompt_name: str = "",  # in the same directory
        llm: str = "default",
        embedding_model: str = "default",
        vector_store: str = "default",
        token_counter: str = "default",
        enable_cache: bool = False,
        cache_path: str = "cache/op",
        sub_ops: Union[List["BaseOp"], Dict[str, "BaseOp"], "BaseOp", None] = None,
        input_mapping: Dict[str, str] | None = None,
        output_mapping: Dict[str, str] | None = None,
        enable_tool_response: bool = False,
        enable_sync_thread_pool: bool = True,
        max_retries: int = 1,
        raise_exception: bool = False,
        **kwargs,
    ):

        super().__init__()

        self.name: str = name or camel_to_snake(self.__class__.__name__)
        self.async_mode: bool = async_mode
        self.language: str = language or C.language
        self.prompt = self._get_prompt_handler(prompt_name)
        self._llm: BaseLLM | str = llm
        self._embedding_model: BaseEmbeddingModel | str = embedding_model
        self._vector_store: BaseVectorStore | str = vector_store
        self._token_counter: BaseTokenCounter | str = token_counter
        self.enable_cache: bool = enable_cache
        self.cache_path: str = cache_path
        self.sub_ops: BaseContext[str, "BaseOp"] = BaseContext[str, BaseOp]()
        self.add_sub_ops(sub_ops)
        self.input_mapping: Dict[str, str] | None = input_mapping
        self.output_mapping: Dict[str, str] | None = output_mapping
        self.enable_tool_response: bool = enable_tool_response
        self.enable_sync_thread_pool: bool = enable_sync_thread_pool
        self.max_retries: int = max(1, max_retries)  # ensure at least 1 retry
        self.raise_exception: bool = raise_exception
        self.op_params: dict = kwargs

        self._pending_tasks: list = []
        self.timer = Timer(name=self.name)
        self.context: RuntimeContext | None = None
        self._cache: CacheHandler | None = None
        self.llm_config: LLMConfig | None = None
        self.embedding_model_config: EmbeddingModelConfig | None = None
        self.vector_store_config: VectorStoreConfig | None = None
        self.token_counter_config: TokenCounterConfig | None = None
        self._tool_call: ToolCall | None = None

    def _get_prompt_handler(self, prompt_name: str) -> PromptHandler:
        file_path: Path = Path(inspect.getfile(self.__class__))
        if prompt_name:
            file_path = file_path.with_stem(prompt_name)
        file_path = file_path.with_suffix(".yaml")
        return PromptHandler(language=self.language).load_prompt_by_file(file_path)

    def _build_tool_call(self) -> ToolCall | None:
        """Build and return the tool call schema for this operator.

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
    def tool_call(self) -> ToolCall:
        """Return the lazily constructed `ToolCall` describing this tool."""
        if self._tool_call is None:
            self._tool_call = self._build_tool_call()
            assert self._tool_call is not None, "tool_call is not defined!"

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
    def input_dict(self) -> dict:
        assert self.tool_call, "tool_call is not defined!"
        input_dict = {}
        for name, attr in self.tool_call.input_schema.items():
            if attr.required:
                assert name in self.context, f"{name} is required!"
                input_dict[name] = self.context[name]
            elif name in self.context:
                input_dict[name] = self.context[name]
        return input_dict

    @property
    def output(self):
        assert self.tool_call, "tool_call is not defined!"
        output_keys = list(self.tool_call.output_schema.keys())
        assert len(output_keys) == 1, "multi output keys is not supported!"
        return self.context[output_keys[0]]

    @output.setter
    def output(self, value):
        assert self.tool_call, "tool_call is not defined!"
        output_keys = list(self.tool_call.output_schema.keys())
        assert len(output_keys) == 1, "multi output keys is not supported!"
        self.context[output_keys[0]] = value

    @property
    def cache(self):
        assert self.enable_cache, "cache is disabled!"
        if self._cache is None:
            self._cache = CacheHandler(f"{self.cache_path}/{self.name}")
        return self._cache

    @property
    def llm(self) -> BaseLLM:
        if isinstance(self._llm, str):
            self.llm_config = C.service_config.llm[self._llm]
            llm_cls = C.get_llm_class(self.llm_config.backend)
            self._llm = llm_cls(model_name=self.llm_config.model_name, **self.llm_config.model_extra)

        return self._llm

    @property
    def embedding_model(self) -> BaseEmbeddingModel:
        if isinstance(self._embedding_model, str):
            self.embedding_model_config = C.service_config.embedding_model[self._embedding_model]
            embedding_model_cls = C.get_embedding_model_class(self.embedding_model_config.backend)
            self._embedding_model = embedding_model_cls(
                model_name=self.embedding_model_config.model_name,
                **self.embedding_model_config.model_extra,
            )

        return self._embedding_model

    @property
    def vector_store(self) -> BaseVectorStore:
        if isinstance(self._vector_store, str):
            self._vector_store = C.get_vector_store(self._vector_store)
        return self._vector_store

    @property
    def token_counter(self) -> BaseTokenCounter:
        if isinstance(self._token_counter, str):
            self.token_counter_config = C.service_config.token_counter[self._token_counter]
            token_counter_cls = C.get_token_counter_class(self.token_counter_config.backend)
            self._token_counter = token_counter_cls(
                model_name=self.token_counter_config.model_name,
                **self.token_counter_config.model_extra,
            )
        return self._token_counter

    @property
    def service_metadata(self) -> dict:
        return C.service_config.model_extra

    @staticmethod
    def build_context(context: RuntimeContext | None = None, **kwargs) -> RuntimeContext:
        if not context:
            context = RuntimeContext()
        if kwargs:
            context.update(kwargs)
        return context

    def before_execute(self):
        if self.input_mapping:
            for name, mapping_name in self.input_mapping.items():
                if name in self.context:
                    self.context[mapping_name] = self.context[name]

        if self.tool_call:
            for name, attrs in self.tool_call.input_schema.items():
                if attrs.required:
                    assert name in self.context, f"{self.name}: {name} is missing in the context!"

    def after_execute(self):
        if self.output_mapping:
            for name, mapping_name in self.output_mapping.items():
                if name in self.context:
                    self.context[mapping_name] = self.context[name]

        if self.tool_call and self.enable_tool_response:
            self.context.response.answer = self.output

    def default_execute(self, e: BaseException):
        if self.tool_call:
            self.output = f"{self.name} failed: {str(e)}"

    def execute_sync(self):
        """all data should flow through context rather than return values"""

    async def execute(self):
        """all data should flow through context rather than return values"""

    def call_sync(self, context: RuntimeContext = None, **kwargs):
        self.context = self.build_context(context, **kwargs)
        with self.timer:
            if self.max_retries == 1 and self.raise_exception:
                self.before_execute()
                self.execute_sync()
                self.after_execute()

            else:
                for i in range(self.max_retries):
                    try:
                        self.before_execute()
                        self.execute_sync()
                        self.after_execute()
                        break

                    except Exception as e:
                        logger.exception(f"{self.name} call_sync failed, error={e.args}")

                        if i == self.max_retries - 1:
                            if self.raise_exception:
                                raise e

                            self.default_execute(e)

        if self.tool_call:
            return self.output
        else:
            return None

    async def call(self, context: RuntimeContext = None, **kwargs) -> Any:
        self.context = self.build_context(context, **kwargs)
        with self.timer:
            if self.max_retries == 1 and self.raise_exception:
                self.before_execute()
                await self.execute()
                self.after_execute()

            else:
                for i in range(self.max_retries):
                    try:
                        self.before_execute()
                        await self.execute()
                        self.after_execute()
                        break

                    except Exception as e:
                        logger.exception(f"{self.name} call failed, error={e.args}")

                        if i == self.max_retries - 1:
                            if self.raise_exception:
                                raise e

                            self.default_execute(e)

        if self.tool_call:
            return self.output
        else:
            return None

    def submit_sync_task(self, fn: Callable, *args, **kwargs) -> "BaseOp":
        if self.enable_sync_thread_pool:
            self._pending_tasks.append(C.thread_pool.submit(fn, *args, **kwargs))
        else:
            self._pending_tasks.append((fn, args, kwargs))
        return self

    def join_sync_tasks(self, task_desc: str = None) -> list:
        result = []
        for task in tqdm(self._pending_tasks, desc=task_desc or self.name):
            if self.enable_sync_thread_pool:
                task_result = task.result()
            else:
                fn, args, kwargs = task
                task_result = fn(*args, **kwargs)

            if task_result:
                if isinstance(task_result, list):
                    result.extend(task_result)
                else:
                    result.append(task_result)

        self._pending_tasks.clear()
        return result

    def submit_async_task(self, fn: Callable, *args, **kwargs):
        assert asyncio.iscoroutinefunction(fn), "fn is not a coroutine function!"
        loop = asyncio.get_running_loop()
        self._pending_tasks.append(loop.create_task(fn(*args, **kwargs)))

    async def join_async_tasks(self, return_exceptions: bool = True):
        result = []

        try:
            task_results = await asyncio.gather(*self._pending_tasks, return_exceptions=return_exceptions)

            for task_result in task_results:
                if return_exceptions and isinstance(task_result, Exception):
                    logger.opt(exception=task_result).error("Task failed with exception")
                    continue

                if task_result:
                    if isinstance(task_result, list):
                        result.extend(task_result)
                    else:
                        result.append(task_result)

        except Exception as e:
            logger.exception(f"join_async_tasks failed with {type(e).__name__}, cancelling remaining tasks...")
            for task in self._pending_tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()
            raise e

        finally:
            self._pending_tasks.clear()

        return result

    def copy(self, **kwargs):
        copy_op = self.__class__(*self._init_args, **self._init_kwargs, **kwargs)
        if self.sub_ops:
            copy_op.sub_ops.clear()
            copy_op.add_sub_ops(self.sub_ops)
        return copy_op

    def add_sub_op(self, op: "BaseOp", name: str = ""):
        assert self.async_mode == op.async_mode, "async mode mismatch!"
        self.sub_ops[name or op.name] = op

    def add_sub_ops(self, sub_ops: Union[List["BaseOp"], Dict[str, "BaseOp"], "BaseOp", None]):
        if sub_ops:
            if isinstance(sub_ops, BaseOp):
                self.add_sub_op(sub_ops)

            elif isinstance(sub_ops, dict):
                for name, op in sub_ops.items():
                    self.add_sub_op(op, name)

            elif isinstance(sub_ops, list):
                for op in sub_ops:
                    self.add_sub_op(op, op.name)

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        return self.prompt.prompt_format(prompt_name=prompt_name, **kwargs)

    def get_prompt(self, prompt_name: str) -> str:
        return self.prompt.get_prompt(prompt_name=prompt_name)

    def __lshift__(self, ops: Union[List["BaseOp"], Dict[str, "BaseOp"], "BaseOp", None]):
        self.add_sub_ops(ops)
        return self

    def __rshift__(self, op: "BaseOp"):
        from .sequential_op import SequentialOp

        sequential_op = SequentialOp(sub_ops=[self], async_mode=self.async_mode)
        if isinstance(op, SequentialOp) and op.sub_ops:
            sequential_op.add_sub_ops(op.sub_ops)
        else:
            sequential_op.add_sub_op(op)
        return sequential_op

    def __or__(self, op: "BaseOp"):
        from .parallel_op import ParallelOp

        parallel_op = ParallelOp(sub_ops=[self], async_mode=self.async_mode)
        if isinstance(op, ParallelOp) and op.sub_ops:
            parallel_op.add_sub_ops(op.sub_ops)
        else:
            parallel_op.add_sub_op(op)
        return parallel_op
