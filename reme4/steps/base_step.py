"""Base step class for LLM workflow execution."""

import copy
from abc import abstractmethod, ABC
from pathlib import Path
from typing import TypeVar, TYPE_CHECKING

from agentscope.formatter import FormatterBase
from agentscope.model import ChatModelBase
from agentscope.token import TokenCounterBase

from ..components.embedding import BaseEmbeddingModel
from ..components.file_parser import BaseFileParser
from ..components.file_store import BaseFileStore
from ..components.file_watcher import BaseFileWatcher
from ..components.prompt_handler import PromptHandler
from ..components.runtime_context import RuntimeContext
from ..enumeration import ComponentEnum
from ..utils import get_logger

if TYPE_CHECKING:
    from ..components import ApplicationContext

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

    @property
    def working_path(self) -> Path:
        """Resolved working directory from app context or cwd."""
        if self.app_context is None:
            return Path.cwd()
        return Path(self.app_context.app_config.working_dir)

    def resolve_path(self, raw: str) -> Path | None:
        """Resolve a relative `path=` argument under self.working_path.

        Rules:
          - the caller supplies the full relative path under ``self.working_path``;
            absolute paths are rejected.
        Returns ``(abs_path, None)`` on success, or ``(None, error_message)`` on failure.
        Filetype-specific gating (e.g. markdown-only / suffix auto-append) is
        layered on top by callers — see ``reme4/steps/crud/_file_io.py::gate_md``.
        """
        if not raw or not str(raw).strip():
            return None, "`path` is required"
        s = str(raw).strip()
        p = Path(s)
        if p.is_absolute():
            return None, (f"path {s!r} is absolute; only relative paths accepted")
        return self.working_path / p, None

    def _resolve(
        self,
        key: str,
        base_cls: type[T],
        comp_enum: ComponentEnum,
        attr: str | None = None,
    ) -> T:
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
