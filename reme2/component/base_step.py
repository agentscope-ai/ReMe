"""Base step class for LLM workflow execution."""

import copy
from abc import abstractmethod

from agentscope.formatter import FormatterBase
from agentscope.model import ChatModelBase
from agentscope.token import TokenCounterBase

from .base_component import BaseComponent
from .chunk_store import BaseChunkStore
from .embedding import BaseEmbeddingModel
from .prompt_handler import PromptHandler
from .runtime_context import RuntimeContext
from ..enumeration import ComponentEnum
from ..schema.file_graph import FileGraph


class BaseStep(BaseComponent):
    """Base step for LLM workflow execution and composition."""

    component_type = ComponentEnum.STEP

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance._init_args = copy.copy(args)
        instance._init_kwargs = copy.copy(kwargs)
        return instance

    def __init__(
            self,
            language: str = "",
            prompt_dict: dict[str, str] | None = None,
            input_mapping: dict[str, str] | None = None,
            output_mapping: dict[str, str] | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.language = language
        self.prompt = PromptHandler(language=self.language)
        self.prompt.load_prompt_by_class(self.__class__).load_prompt_dict(prompt_dict)
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.context: RuntimeContext | None = None

    @abstractmethod
    async def execute(self):
        """Execute the step logic."""

    async def __call__(self, context: RuntimeContext | None = None, **kwargs):
        self.context = RuntimeContext.from_context(context, **kwargs)
        assert self.context is not None

        if self.input_mapping:
            self.context.apply_mapping(self.input_mapping)

        result = await self.execute()

        if self.output_mapping:
            self.context.apply_mapping(self.output_mapping)

        return result

    def _get_component(self, key: ComponentEnum, name: str, attr: str | None = None):
        assert self.app_context is not None
        comp = self.app_context.components[key][name]
        return getattr(comp, attr) if attr else comp

    @property
    def as_llm(self) -> ChatModelBase:
        name = self.kwargs.get("as_llm", "default")
        return name if isinstance(name, ChatModelBase) else self._get_component(ComponentEnum.AS_LLM, name, "model")

    @property
    def as_llm_formatter(self) -> FormatterBase:
        name = self.kwargs.get("as_llm_formatter", "default")
        return name if isinstance(name, FormatterBase) else self._get_component(ComponentEnum.AS_LLM_FORMATTER, name,
                                                                                "formatter")

    @property
    def as_token_counter(self) -> TokenCounterBase:
        name = self.kwargs.get("as_token_counter", "default")
        return name if isinstance(name, TokenCounterBase) else self._get_component(ComponentEnum.AS_TOKEN_COUNTER, name,
                                                                                   "token_counter")

    @property
    def chunk_store(self) -> BaseChunkStore:
        name = self.kwargs.get("chunk_store", "default")
        return name if isinstance(name, BaseChunkStore) else self._get_component(ComponentEnum.CHUNK_STORE, name)

    @property
    def file_graph(self) -> FileGraph:
        name = self.kwargs.get("file_watcher", "default")
        watcher = self._get_component(ComponentEnum.FILE_WATCHER, name)
        return watcher.file_graph

    @property
    def embedding(self) -> BaseEmbeddingModel:
        name = self.kwargs.get("embedding", "default")
        return name if isinstance(name, BaseEmbeddingModel) else self._get_component(ComponentEnum.EMBEDDING_MODEL,
                                                                                     name)

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        return self.prompt.prompt_format(prompt_name=prompt_name, **kwargs)

    def get_prompt(self, prompt_name: str) -> str:
        return self.prompt.get_prompt(prompt_name=prompt_name)

    def copy(self, **kwargs) -> "BaseStep":
        return self.__class__(*self._init_args, **{**self._init_kwargs, **kwargs})
