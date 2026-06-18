"""Base job component for sequential step execution."""

from typing import TYPE_CHECKING

from ..base_component import BaseComponent
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ComponentEnum
from ...schema import ComponentConfig, Response

if TYPE_CHECKING:
    from ...steps import BaseStep


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.description = description
        self.parameters = parameters or {}
        self.step_configs = steps or []
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
        step_cls = R.get(ComponentEnum.STEP, config.backend)
        if not step_cls:
            raise ValueError(f"Unregistered backend '{config.backend}' of type '{ComponentEnum.STEP}'")
        params = config.model_dump()
        params["app_context"] = self.app_context
        return step_cls, params

    def _build_steps(self) -> list["BaseStep"]:
        # dict(params) copies kwargs so steps cannot mutate the shared spec.
        return [step_cls(**dict(params)) for step_cls, params in self.step_specs]

    async def __call__(self, **kwargs) -> Response:
        """Run all steps in order, capturing any failure into the response."""
        merged = {**self.kwargs, **kwargs}
        context = RuntimeContext(**merged)
        try:
            for step in self._build_steps():
                await step(context)
        except Exception as e:
            self.logger.exception(f"Failed to execute job: {e}")
            context.response.success = False
            context.response.answer = str(e)
        return context.response
