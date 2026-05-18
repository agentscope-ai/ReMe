"""Base step class for LLM workflow execution."""

import copy
from abc import abstractmethod, ABC
from typing import TypeVar, TYPE_CHECKING

from agentscope.formatter import FormatterBase
from agentscope.message import TextBlock
from agentscope.model import ChatModelBase
from agentscope.token import TokenCounterBase
from agentscope.tool import Toolkit, ToolResponse

from ..components.embedding import BaseEmbeddingModel
from ..components.file_parser import BaseFileParser
from ..components.file_store import BaseFileStore
from ..components.file_watcher import BaseFileWatcher
from ..components.prompt_handler import PromptHandler
from ..components.runtime_context import RuntimeContext
from ..enumeration import ComponentEnum
from ..schema import Response
from ..utils import get_logger

if TYPE_CHECKING:
    from ..components import ApplicationContext
    from ..components.job import BaseJob

T = TypeVar("T")


class BaseStep(ABC):
    """Composable unit of an LLM workflow."""

    component_type = ComponentEnum.STEP

    def __new__(cls, *args, **kwargs):
        # Snapshot init args so copy() can rebuild an equivalent instance later.
        instance = object.__new__(cls)
        instance._init_args = copy.copy(args)
        instance._init_kwargs = copy.copy(kwargs)
        return instance

    def __init__(
        self,
        name: str | None = None,
        backend: str = "",
        app_context: "ApplicationContext | None" = None,
        language: str = "",
        prompt_dict: dict[str, str] | None = None,
        input_mapping: dict[str, str] | None = None,
        output_mapping: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.name: str = name or self.__class__.__name__
        self.backend: str = backend
        self.app_context: "ApplicationContext | None" = app_context
        self.language: str = language
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.kwargs: dict = kwargs
        self.context: RuntimeContext | None = None

        self.logger = get_logger()
        if hasattr(self.logger, "bind"):
            self.logger = self.logger.bind(component=self.name)

        # Load class-level prompts first, then overlay caller-provided overrides.
        self.prompt = PromptHandler(language=self.language)
        self.prompt.load_prompt_by_class(self.__class__).load_prompt_dict(prompt_dict)

    @abstractmethod
    async def execute(self):
        """Run the step's logic against ``self.context``."""

    async def __call__(self, context: RuntimeContext | None = None, **kwargs):
        # Build runtime context, then apply key remapping around execute().
        self.context = RuntimeContext.from_context(context, **kwargs)
        assert self.context is not None
        if self.input_mapping:
            self.context.apply_mapping(self.input_mapping)
        result = await self.execute()
        if self.output_mapping:
            self.context.apply_mapping(self.output_mapping)
        return result

    def _resolve(self, key: str, base_cls: type[T], comp_enum: ComponentEnum, attr: str | None = None) -> T:
        """Return a kwargs-supplied instance, or look one up by name in the app registry."""
        # 1. Step init kwargs, 2. Runtime context (run_job kwargs), 3. App registry by name.
        for source in (self.kwargs, self.context or {}):
            value = source.get(key)
            if isinstance(value, base_cls):
                return value

        name = self.kwargs.get(key, "default")
        assert self.app_context is not None
        comp = self.app_context.components[comp_enum][name]
        return getattr(comp, attr) if attr else comp

    @property
    def as_llm(self) -> ChatModelBase:
        """Return the chat model component."""
        return self._resolve("as_llm", ChatModelBase, ComponentEnum.AS_LLM, "model")

    @property
    def as_llm_formatter(self) -> FormatterBase:
        """Return the LLM formatter component."""
        return self._resolve("as_llm_formatter", FormatterBase, ComponentEnum.AS_LLM_FORMATTER, "formatter")

    @property
    def as_token_counter(self) -> TokenCounterBase:
        """Return the token counter component."""
        return self._resolve("as_token_counter", TokenCounterBase, ComponentEnum.AS_TOKEN_COUNTER, "token_counter")

    @property
    def file_parser(self) -> BaseFileParser:
        """Return the file parser component."""
        return self._resolve("file_parser", BaseFileParser, ComponentEnum.FILE_PARSER)

    @property
    def file_store(self) -> BaseFileStore:
        """Return the file store component."""
        return self._resolve("file_store", BaseFileStore, ComponentEnum.FILE_STORE)

    @property
    def embedding(self) -> BaseEmbeddingModel:
        """Return the embedding model component."""
        return self._resolve("embedding", BaseEmbeddingModel, ComponentEnum.EMBEDDING_MODEL)

    @property
    def file_watcher(self) -> BaseFileWatcher:
        """Return the file watcher component."""
        return self._resolve("file_watcher", BaseFileWatcher, ComponentEnum.FILE_WATCHER)

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        """Format a named prompt template with the given kwargs."""
        return self.prompt.prompt_format(prompt_name=prompt_name, **kwargs)

    def get_prompt(self, prompt_name: str) -> str:
        """Return a named prompt template as-is."""
        return self.prompt.get_prompt(prompt_name=prompt_name)

    def copy(self, **kwargs) -> "BaseStep":
        """Construct a new instance from the original init args, applying overrides."""
        return self.__class__(*self._init_args, **{**self._init_kwargs, **kwargs})

    def get_job(self, name: str) -> "BaseJob | None":
        """Return a job by name."""
        if self.app_context is None:
            raise RuntimeError("Cannot get job without an app context")
        return self.app_context.jobs.get(name)

    async def run_job(self, name: str, **kwargs) -> Response:
        """Execute a job by name and kwargs, return the final response."""
        job: "BaseJob | None" = self.get_job(name)
        if job is None:
            raise RuntimeError(f"Job {name} not found")
        return await job(**kwargs)

    def add_as_tool(self, toolkit: Toolkit, job_name: str) -> None:
        job: "BaseJob | None" = self.get_job(job_name)
        if job is None:
            raise RuntimeError(f"Job {job_name} not found")

        async def run_job(**kwargs) -> ToolResponse:
            response = await job(**kwargs)
            return ToolResponse(content=[TextBlock(type="text", text=response.answer)])

        toolkit.register_tool_function(
            tool_func=run_job,
            func_name=job_name,
            func_description=job.description,
            json_schema=job.parameters,
        )
