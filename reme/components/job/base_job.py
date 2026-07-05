"""Base job component for sequential step execution."""

from typing import TYPE_CHECKING

from ..base_component import BaseComponent
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ComponentEnum
from ...schema import ComponentConfig, Response

if TYPE_CHECKING:
    from ...steps import BaseStep

_LOCAL_INSTANTIATION_KEY = "_local_instantiation_"


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
        self.step_specs: list[tuple[type["BaseStep"], dict, int]] = []
        self._persistent_steps: dict[int, "BaseStep"] = {}

    async def _start(self) -> None:
        if self.app_context is None:
            raise RuntimeError(f"app_context must be provided for job '{self.name}'")
        self.step_specs = [self._resolve_step(raw) for raw in self.step_configs]
        self._instantiate_persistent_steps()

    def _instantiate_persistent_steps(self) -> None:
        """Instantiate steps with _local_instantiation_ > 0, grouping by value."""
        self._persistent_steps = {}
        for step_cls, params, local_id in self.step_specs:
            # Only strictly positive integers indicate persistent lifecycle.
            if isinstance(local_id, int) and local_id > 0 and local_id not in self._persistent_steps:
                self._persistent_steps[local_id] = step_cls(**dict(params))

    async def _close(self) -> None:
        self.step_specs.clear()
        self._persistent_steps.clear()

    def _resolve_step(self, raw: ComponentConfig | dict) -> tuple[type["BaseStep"], dict, int]:
        """Validate a step config and look up its class via the registry."""
        config = raw if isinstance(raw, ComponentConfig) else ComponentConfig(**raw)
        if not config.backend:
            raise ValueError("Step is missing the required 'backend' field")
        step_cls = R.get(ComponentEnum.STEP, config.backend)
        if not step_cls:
            raise ValueError(f"Unregistered backend '{config.backend}' of type '{ComponentEnum.STEP}'")
        params = config.model_dump()
        params["app_context"] = self.app_context
        # Extract and strip the lifecycle marker before passing to step constructor.
        local_id = int(params.pop(_LOCAL_INSTANTIATION_KEY, 0))
        return step_cls, params, local_id

    def _build_steps(self) -> list["BaseStep"]:
        """Build the step list: persistent steps reuse existing instances."""
        steps: list["BaseStep"] = []
        for step_cls, params, local_id in self.step_specs:
            # Only strictly positive integers indicate persistent lifecycle.
            if isinstance(local_id, int) and local_id > 0:
                steps.append(self._persistent_steps[local_id])
            else:
                steps.append(step_cls(**dict(params)))
        return steps

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
