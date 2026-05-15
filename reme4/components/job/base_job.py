"""Base job component for sequential step execution."""

from ..base_component import BaseComponent
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ComponentEnum
from ...schema import ComponentConfig, Response


@R.register("base")
class BaseJob(BaseComponent):
    """Job that executes steps sequentially and returns a Response."""

    component_type = ComponentEnum.JOB

    def __init__(self, description: str, parameters: dict, steps: list[ComponentConfig | dict], **kwargs):
        super().__init__(**kwargs)
        self.description = description
        self.parameters = parameters or {}
        self.step_configs = steps or []

        from ...steps import BaseStep

        self.step_components: list[BaseStep] = []

    async def _start(self) -> None:
        """Resolve step configs into instantiated step components."""
        assert self.app_context is not None, "app_context must be provided"
        for raw in self.step_configs:
            config = raw if isinstance(raw, ComponentConfig) else ComponentConfig(**raw)
            if not config.backend:
                raise ValueError("Step is missing the required 'backend' field")
            step_cls = R.get(ComponentEnum.STEP, config.backend)
            if not step_cls:
                raise ValueError(f"Unregistered backend '{config.backend}' of type '{ComponentEnum.STEP}'")
            params = config.model_dump()
            params["app_context"] = self.app_context
            self.step_components.append(step_cls(**params))

    async def _close(self) -> None:
        """Release all step components."""
        self.step_components.clear()

    async def __call__(self, **kwargs) -> Response:
        """Execute all steps in order and return the final response."""
        context = RuntimeContext(**kwargs)
        try:
            for step in self.step_components:
                await step(context)
        except Exception as e:
            self.logger.exception(f"Failed to execute job: {e}")
            context.response.answer = str(e)
        return context.response
