"""Base operator class for LLM workflow execution and composition."""

import asyncio
import copy
import inspect
from abc import ABCMeta
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from ..component import PromptHandler
from ..component import RuntimeContext
from ..component.application_context import ApplicationContext
from ..component.embedding import BaseEmbeddingModel
from ..component.file_store import BaseFileStore
from ..schema import ApplicationConfig
from ..enumeration import ComponentEnum
if TYPE_CHECKING:
    from ..component.as_llm import AsOpenAIChatModel


class BaseOp(metaclass=ABCMeta):
    """Base operator class for LLM workflow execution and composition."""

    def __new__(cls, *args, **kwargs):
        """Capture initialization arguments for object cloning."""
        instance = super().__new__(cls)
        instance._init_args = copy.copy(args)
        instance._init_kwargs = copy.copy(kwargs)
        return instance

    def __init__(
            self,
            name: str = "",
            language: str = "",
            prompt_dict: dict[str, str] | None = None,
            input_mapping: dict[str, str] | None = None,
            output_mapping: dict[str, str] | None = None,
            **kwargs,
    ):
        """Initialize operator configurations and internal state."""
        self.name = name or self.__class__.__name__
        self.language = language
        self.prompt = PromptHandler(language=self.language)
        self.prompt.load_prompt_by_file(Path(inspect.getfile(self.__class__)).with_suffix(".yaml")) \
            .load_prompt_dict(prompt_dict)

        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.enable_parallel = enable_parallel
        self.max_retries = max(1, max_retries)
        self.raise_exception = raise_exception
        self.op_params = kwargs

        self._pending_tasks: list = []
        self.context: RuntimeContext | None = None

    async def before_execute(self):
        self.context.apply_mapping(self.input_mapping)

    async def execute(self):
        """"""

    async def after_execute(self, output):
        self.context.apply_mapping(self.output_mapping)
        return output

    async def call(self, context: RuntimeContext = None, **kwargs):
        self.context = RuntimeContext.from_context(context, **kwargs)
        await self.before_execute()
        response = await self.execute()
        response = await self.after_execute(response)
        return response

    @property
    def application_context(self) -> ApplicationContext:
        return self.context.application_context

    @property
    def app_config(self) -> ApplicationConfig:
        return self.application_context.app_config

    @property
    def as_llm(self) -> ChatModelBase:
        """Get the AgentScope LLM instance from ServiceContext."""
        as_llm_name = self.op_params.get("as_llm", "default")
        as_llm_dict = self.application_context.components[ComponentEnum.AS_LLM]

        return as_llm_dict[as_llm_name]

    @property
    def as_llm_formatter(self) -> FormatterBase:
        """Get the AgentScope LLM formatter instance from ServiceContext."""
        if isinstance(self._as_llm_formatter, str):
            self._as_llm_formatter = self.service_context.as_llm_formatters[self._as_llm_formatter]
        return self._as_llm_formatter

    @property
    def as_token_counter(self) -> HuggingFaceTokenCounter:
        """Get the token counter instance from ServiceContext."""
        if isinstance(self._as_token_counter, str):
            self._as_token_counter = self.service_context.as_token_counters[self._as_token_counter]
        return self._as_token_counter

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

    def submit_async_task(self, coro_fn: Callable, *args, **kwargs) -> "BaseOp":
        """Submit an async task to the pending tasks queue."""
        task = coro_fn(*args, **kwargs)
        self._pending_tasks.append(task)
        return self

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

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        """Format a prompt template with provided keyword arguments."""
        return self.prompt.prompt_format(prompt_name=prompt_name, **kwargs)

    def get_prompt(self, prompt_name: str) -> str:
        """Get a prompt template by name."""
        return self.prompt.get_prompt(prompt_name=prompt_name)

    def copy(self, **kwargs):
        """Create a copy of this operator with optional parameter overrides."""
        return self.__class__(*self._init_args, **{**self._init_kwargs, **kwargs})
