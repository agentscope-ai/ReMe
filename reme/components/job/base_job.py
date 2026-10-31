"""Base job component for sequential step execution."""

from typing import TYPE_CHECKING

from ..base_component import BaseComponent
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ComponentEnum
from ...schema import ComponentConfig, Response
from ...utils import global_counter_inc

if TYPE_CHECKING:
    from ...steps import BaseStep


def _exception_chain(exc: BaseException):
    """Walk explicit causes or unsuppressed contexts without following cycles."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None


def _describe_exception(exc: BaseException) -> str:
    """Render an exception and its causes without dropping empty messages."""
    parts = []
    for current in _exception_chain(exc):
        message = str(current).strip()
        name = type(current).__name__
        parts.append(f"{name}: {message}" if message else name)
    return " <- ".join(parts)


def _exception_details(exc: BaseException) -> dict:
    """Expose classification fields only, without requests, headers or bodies."""
    chain = []
    for current in _exception_chain(exc):
        details = {"type": type(current).__name__, "module": type(current).__module__}
        status = getattr(current, "status_code", None)
        if not isinstance(status, int):
            status = getattr(getattr(current, "response", None), "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            details["status_code"] = status
        chain.append(details)
    return {**chain[0], "causes": chain[1:]}


@R.register("base")
class BaseJob(BaseComponent):
    """Job that executes steps sequentially and returns a Response."""

    component_type = ComponentEnum.JOB

    def __init__(
        self,
        description: str = "",
        parameters: dict | None = None,
        steps: list[ComponentConfig | dict] | None = None,
        enable_serve: bool = True,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.description = description
        self.parameters = parameters or {}
        self.step_configs = steps or []
        self.enabled = enabled
        self.enable_serve = enable_serve
        self.step_specs: list[tuple[type["BaseStep"], dict]] = []

    async def _start(self) -> None:
        if self.app_context is None:
            raise RuntimeError(f"app_context must be provided for job '{self.name}'")
        self.step_specs = [self._resolve_step(raw) for raw in self.step_configs]

    async def _close(self) -> None:
        self.step_specs.clear()

    def _resolve_step(self, raw: ComponentConfig | dict) -> tuple[type["BaseStep"], dict]:
        """Validate a step config and look up its class via the registry."""
        config = raw if isinstance(raw, ComponentConfig) else ComponentConfig(**raw)
        if not config.backend:
            raise ValueError("Step is missing the required 'backend' field")
        step_cls = self.app_context.registry.get(ComponentEnum.STEP, config.backend)
        if not step_cls:
            raise ValueError(f"Unregistered backend '{config.backend}' of type '{ComponentEnum.STEP}'")
        params = config.model_dump()
        params["app_context"] = self.app_context
        return step_cls, params

    def _build_steps(self) -> list["BaseStep"]:
        # dict(params) copies kwargs so steps cannot mutate the shared spec.
        return [step_cls(**dict(params)) for step_cls, params in self.step_specs]

    def _record_call(self) -> None:
        """Increment this job's application-lifetime call counter."""
        metadata = getattr(self.app_context, "metadata", None)
        if isinstance(metadata, dict):
            global_counter_inc(metadata, ["__job_counter", self.name])

    def check_enabled(self) -> None:
        """Reject execution of a disabled job."""
        if not self.enabled:
            raise ValueError(f"Job '{self.name}' is disabled")

    async def __call__(self, **kwargs) -> Response:
        """Run all steps in order, capturing any failure into the response."""
        self.check_enabled()
        self._record_call()
        merged = {**self.kwargs, **kwargs}
        context = RuntimeContext(**merged)
        try:
            for step in self._build_steps():
                await step(context)
        except Exception as e:
            self.logger.exception(f"Failed to execute job: {e}")
            context.response.success = False
            context.response.answer = _describe_exception(e)
            context.response.metadata["reme.job_error"] = _exception_details(e)
        return context.response
