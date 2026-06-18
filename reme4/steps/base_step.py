"""Base step class for LLM workflow execution."""

import copy
from abc import abstractmethod, ABC
from typing import Any, TYPE_CHECKING

from agentscope.model import ChatModelBase

from ..components.agent_wrapper.base_agent_wrapper import BaseAgentWrapper
from ..components.base_component import ComponentMixin
from ..components.component_registry import R
from ..components.file_catalog import BaseFileCatalog
from ..components.file_store import BaseFileStore
from ..components.prompt_handler import PromptHandler
from ..components.runtime_context import RuntimeContext
from ..enumeration import ComponentEnum
from ..schema import ApplicationConfig, Response

if TYPE_CHECKING:
    from ..components import ApplicationContext
    from ..components.job import BaseJob

_UNSET = object()
_DispatchStep = str | dict[str, Any]


class Ref:
    """Descriptor that lazily resolves a component dependency for Steps.

    Replaces the ``@property`` + ``_resolve()`` boilerplate with a single
    class-level declaration::

        as_llm = Ref(ChatModelBase, ComponentEnum.AS_LLM, "model")
        file_store = Ref(BaseFileStore, ComponentEnum.FILE_STORE)

    Resolution follows a 3-source fallback identical to the old ``_resolve``:
    ``kwargs`` -> ``context`` -> ``app_context`` component registry.
    The resolved value is cached on the instance for its lifetime
    (steps are rebuilt per job call via ``_build_steps``).
    """

    __slots__ = ("base_cls", "comp_enum", "attr", "optional", "key", "_cache_attr")

    def __init__(self, base_cls: type, comp_enum: ComponentEnum, attr: str | None = None, *, optional: bool = False):
        self.base_cls = base_cls
        self.comp_enum = comp_enum
        self.attr = attr
        self.optional = optional
        self.key: str = ""
        self._cache_attr: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.key = name
        self._cache_attr = f"_ref_{name}"

    def __get__(self, obj: "BaseStep | None", objtype: type | None = None):
        if obj is None:
            return self
        cached = obj.__dict__.get(self._cache_attr, _UNSET)
        if cached is not _UNSET:
            return cached
        value = self._resolve(obj)
        obj.__dict__[self._cache_attr] = value
        return value

    def __set__(self, obj: "BaseStep", value) -> None:
        obj.__dict__[self._cache_attr] = value

    def __delete__(self, obj: "BaseStep") -> None:
        obj.__dict__.pop(self._cache_attr, None)

    def _resolve(self, obj: "BaseStep"):
        for source in (obj.kwargs, obj.context or {}):
            value = source.get(self.key)
            if isinstance(value, self.base_cls):
                return value

        name = obj.kwargs.get(self.key, "default")
        if obj.app_context is None:
            if self.optional:
                return None
            raise RuntimeError(f"app_context is not set when resolving '{self.key}'")
        comp = obj.app_context.components[self.comp_enum].get(name)
        if comp is None:
            if self.optional:
                return None
            raise KeyError(f"Component '{name}' not found in {self.comp_enum.value}")
        return getattr(comp, self.attr) if self.attr else comp


class BaseStep(ComponentMixin, ABC):
    """Composable unit of an LLM workflow."""

    component_type = ComponentEnum.STEP

    as_llm: ChatModelBase = Ref(ChatModelBase, ComponentEnum.AS_LLM, "model")
    agent_wrapper: BaseAgentWrapper = Ref(BaseAgentWrapper, ComponentEnum.AGENT_WRAPPER, optional=True)
    file_catalog: BaseFileCatalog = Ref(BaseFileCatalog, ComponentEnum.FILE_CATALOG, optional=True)
    file_store: BaseFileStore = Ref(BaseFileStore, ComponentEnum.FILE_STORE)

    def __new__(cls, *args, **kwargs):
        # Snapshot init args so copy() can rebuild an equivalent instance later.
        instance = object.__new__(cls)
        instance._init_args, instance._init_kwargs = copy.copy(args), copy.copy(kwargs)
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
        dispatch_steps: list[_DispatchStep] | None = None,
        **kwargs,
    ):
        super().__init__(name=name, backend=backend, app_context=app_context, **kwargs)
        self.language = language or (self.app_context.app_config.language if self.app_context is not None else "")
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.dispatch_step_specs = list(dispatch_steps or [])
        self.context: RuntimeContext | None = None

        # Load class-level prompts first, then overlay caller-provided overrides.
        # Walk MRO in reverse so most-derived class wins; subclasses without their
        # own YAML inherit prompts from their parent (e.g. AutoDreamStep inherits
        # dream.yaml from DreamStep).
        self.prompt = PromptHandler(language=self.language)
        for cls in reversed(self.__class__.__mro__):
            self.prompt.load_prompt_by_class(cls)
        self.prompt.load_prompt_dict(prompt_dict)

    @abstractmethod
    async def execute(self):
        """Run the step's logic against ``self.context``."""

    async def __call__(self, context: RuntimeContext | None = None, **kwargs):
        # Clear cached Ref values so context-supplied overrides take effect.
        for key in [k for k in self.__dict__ if k.startswith("_ref_")]:
            del self.__dict__[key]
        self.context = RuntimeContext.from_context(context, **kwargs)
        assert self.context is not None
        if self.input_mapping:
            self.context.apply_mapping(self.input_mapping)
        result = await self.execute()
        if self.output_mapping:
            self.context.apply_mapping(self.output_mapping)
        return result

    def prompt_format(self, prompt_name: str, **kwargs) -> str:
        """Format a named prompt template with the given kwargs."""
        return self.prompt.prompt_format(prompt_name=prompt_name, **kwargs)

    def get_prompt(self, prompt_name: str) -> str:
        """Return a named prompt template as-is."""
        return self.prompt.get_prompt(prompt_name=prompt_name)

    def config_value(self, key: str):
        """Return an app config value, falling back to ApplicationConfig defaults."""
        defaults = ApplicationConfig()
        cfg = self.app_context.app_config if self.app_context is not None else defaults
        value = getattr(cfg, key)
        return getattr(defaults, key) if value in (None, "") else value

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

    def _resolve_dispatch_step(self, raw: _DispatchStep):
        """Resolve a dispatch step spec to (step class, init params)."""
        if isinstance(raw, str):
            params: dict[str, Any] = {"backend": raw}
        elif isinstance(raw, dict):
            params = dict(raw)
        else:
            raise TypeError(f"Invalid dispatch step spec: {raw!r}")

        backend = params.get("backend", "")
        if not backend:
            raise ValueError("Dispatch step is missing the required 'backend' field")
        step_cls = R.get(ComponentEnum.STEP, backend)
        if step_cls is None:
            raise RuntimeError(f"Unregistered step '{backend}'")
        params["app_context"] = self.app_context
        return step_cls, params

    async def dispatch_steps(self, dispatch_steps: list[_DispatchStep], **kwargs) -> list[Response]:
        """Run dispatch steps against the current context.

        Callers pass producer-specific values, usually ``changes=...``. Existing
        context data is preserved for downstream handlers.
        """
        if self.context is None:
            raise RuntimeError("Cannot dispatch steps without a runtime context")

        responses: list[Response] = []
        for raw in dispatch_steps:
            step_cls, params = self._resolve_dispatch_step(raw)
            responses.append(await step_cls(**params)(self.context, **kwargs))
        return responses
