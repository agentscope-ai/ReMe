"""Base step class for LLM workflow execution."""

import copy
from abc import abstractmethod

from agentscope.formatter import FormatterBase
from agentscope.model import ChatModelBase
from agentscope.token import TokenCounterBase

from .base_component import BaseComponent
from .file_store import BaseFileStore
from .embedding import BaseEmbeddingModel
from .prompt_handler import PromptHandler
from .runtime_context import RuntimeContext
from ..enumeration import ComponentEnum


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

    def get_component(self, key: ComponentEnum, name: str, attr: str | None = None):
        assert self.app_context is not None
        comp = self.app_context.components[key][name]
        return getattr(comp, attr) if attr else comp

    def _get_component_optional(self, key: ComponentEnum, name: str = "default", attr: str | None = None):
        """Like `_get_component` but returns None instead of raising when
        the component / attribute is missing. For features that should
        gracefully degrade when an LLM (etc.) isn't configured."""
        if self.app_context is None:
            return None
        comp = self.app_context.components.get(key, {}).get(name)
        if comp is None:
            return None
        return getattr(comp, attr, None) if attr else comp

    @property
    def as_llm(self) -> ChatModelBase:
        name = self.kwargs.get("as_llm", "default")
        return name if isinstance(name, ChatModelBase) else self.get_component(ComponentEnum.AS_LLM, name, "model")

    @property
    def as_llm_formatter(self) -> FormatterBase:
        name = self.kwargs.get("as_llm_formatter", "default")
        return (
            name
            if isinstance(name, FormatterBase)
            else self.get_component(
                ComponentEnum.AS_LLM_FORMATTER,
                name,
                "formatter",
            )
        )

    @property
    def as_token_counter(self) -> TokenCounterBase:
        name = self.kwargs.get("as_token_counter", "default")
        return (
            name
            if isinstance(name, TokenCounterBase)
            else self.get_component(
                ComponentEnum.AS_TOKEN_COUNTER,
                name,
                "token_counter",
            )
        )

    @property
    def file_store(self) -> BaseFileStore:
        name = self.kwargs.get("file_store", "default")
        return name if isinstance(name, BaseFileStore) else self.get_component(ComponentEnum.FILE_STORE, name)

    @property
    def embedding(self) -> BaseEmbeddingModel:
        name = self.kwargs.get("embedding", "default")
        return (
            name
            if isinstance(name, BaseEmbeddingModel)
            else self.get_component(
                ComponentEnum.EMBEDDING_MODEL,
                name,
            )
        )

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        return self.prompt.prompt_format(prompt_name=prompt_name, **kwargs)

    def get_prompt(self, prompt_name: str) -> str:
        return self.prompt.get_prompt(prompt_name=prompt_name)

    def copy(self, **kwargs) -> "BaseStep":
        return self.__class__(*self._init_args, **{**self._init_kwargs, **kwargs})
