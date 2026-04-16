"""Base step class for LLM workflow execution."""

import copy
from abc import abstractmethod

from .application_context import ApplicationContext
from .as_llm import BaseAsLLM
from .as_llm_formatter import BaseAsLLMFormatter
from .as_token_counter import BaseAsTokenCounter
from .base_component import BaseComponent
from .embedding import BaseEmbeddingModel
from .file_store import BaseFileStore
from .prompt_handler import PromptHandler
from .runtime_context import RuntimeContext
from ..enumeration import ComponentEnum
from ..schema import ApplicationConfig
from ..utils import camel_to_snake


class BaseStep(BaseComponent):
    """Base step for LLM workflow execution and composition."""

    component_type = ComponentEnum.STEP

    def __new__(cls, *args, **kwargs):
        """Capture init args for object cloning."""
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
        """Initialize step configurations."""
        super().__init__(**kwargs)
        self.name = name or camel_to_snake(self.__class__.__name__)
        self.language = language
        self.prompt = PromptHandler(language=self.language)
        self.prompt.load_prompt_by_class(self.__class__).load_prompt_dict(prompt_dict)
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.context: RuntimeContext | None = None

    async def _start(self, app_context=None) -> None:
        """Apply input mapping before execution."""
        if self.input_mapping and self.context:
            self.context.apply_mapping(self.input_mapping)

    async def _close(self) -> None:
        """Apply output mapping after execution."""
        if self.output_mapping and self.context:
            self.context.apply_mapping(self.output_mapping)

    @abstractmethod
    async def execute(self):
        """Execute the step logic."""

    async def __call__(self, context: RuntimeContext | None = None, **kwargs):
        """Execute the step with lifecycle management."""
        self.context = RuntimeContext.from_context(context, **kwargs)
        await self.start()
        try:
            response = await self.execute()
            return response
        finally:
            await self.close()

    @property
    def application_context(self) -> ApplicationContext:
        """Get the application context from runtime context."""
        assert self.context is not None, "Runtime context not set."
        return self.context.application_context

    @property
    def app_config(self) -> ApplicationConfig:
        """Get the application configuration."""
        return self.application_context.app_config

    @property
    def as_llm(self) -> BaseAsLLM:
        """Get the AsLLM instance by name."""
        name: str = self.kwargs.get("as_llm", "default")
        llms = self.application_context.components[ComponentEnum.AS_LLM]
        if name not in llms:
            raise ValueError(f"AsLLM {name} not found.")
        llm = llms[name]
        if not isinstance(llm, BaseAsLLM):
            raise TypeError(f"{name} is not a BaseAsLLM instance.")
        return llm

    @property
    def as_llm_formatter(self) -> BaseAsLLMFormatter:
        """Get the AsLLMFormatter instance by name."""
        name: str = self.kwargs.get("as_llm_formatter", "default")
        formatters = self.application_context.components[ComponentEnum.AS_LLM_FORMATTER]
        if name not in formatters:
            raise ValueError(f"AsLLMFormatter {name} not found.")
        formatter = formatters[name]
        if not isinstance(formatter, BaseAsLLMFormatter):
            raise TypeError(f"{name} is not a BaseAsLLMFormatter instance.")
        return formatter

    @property
    def as_token_counter(self):
        """Get the TokenCounter instance by name."""
        name: str = self.kwargs.get("as_token_counter", "default")
        counters = self.application_context.components[ComponentEnum.AS_TOKEN_COUNTER]
        if name not in counters:
            raise ValueError(f"AsTokenCounter {name} not found.")
        counter = counters[name]
        if not isinstance(counter, BaseAsTokenCounter):
            raise TypeError(f"{name} is not a BaseAsTokenCounter instance.")
        return counter

    @property
    def file_store(self) -> BaseFileStore:
        """Get the FileStore instance by name."""
        name: str = self.kwargs.get("file_store", "default")
        stores = self.application_context.components[ComponentEnum.FILE_STORE]
        if name not in stores:
            raise ValueError(f"FileStore {name} not found.")
        store = stores[name]
        if not isinstance(store, BaseFileStore):
            raise TypeError(f"{name} is not a BaseFileStore instance.")
        return store

    @property
    def embedding(self) -> BaseEmbeddingModel:
        """Get the EmbeddingModel instance by name."""
        name: str = self.kwargs.get("embedding", "default")
        models = self.application_context.components[ComponentEnum.EMBEDDING_MODEL]
        if name not in models:
            raise ValueError(f"EmbeddingModel {name} not found.")
        model = models[name]
        if not isinstance(model, BaseEmbeddingModel):
            raise TypeError(f"{name} is not a BaseEmbeddingModel instance.")
        return model

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        """Format a prompt template."""
        return self.prompt.prompt_format(prompt_name=prompt_name, **kwargs)

    def get_prompt(self, prompt_name: str) -> str:
        """Get a prompt template by name."""
        return self.prompt.get_prompt(prompt_name=prompt_name)

    def copy(self, **kwargs) -> "BaseStep":
        """Create a copy with optional parameter overrides."""
        return self.__class__(*self._init_args, **{**self._init_kwargs, **kwargs})
