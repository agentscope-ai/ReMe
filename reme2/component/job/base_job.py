"""Base job component for sequential step execution."""
from typing import TYPE_CHECKING

from ..base_component import BaseComponent
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ComponentEnum
from ...schema import Response, ComponentConfig

if TYPE_CHECKING:
    from ..base_step import BaseStep


@R.register("base")
class BaseJob(BaseComponent):
    """Job that executes steps sequentially."""

    component_type = ComponentEnum.JOB

    def __init__(
            self,
            description: str = "",
            parameters: dict | None = None,
            steps: list[ComponentConfig] | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.description = description
        self.parameters = parameters or {}
        self.step_configs = steps or []
        self.step_components: list["BaseStep"] = []

    async def _start(self) -> None:
        assert self.app_context is not None, "app_context must be provided"
        for config in self.step_configs:
            if not config.backend:
                raise ValueError(f"Step is missing the required 'backend' field")
            step_cls = R.get(ComponentEnum.STEP, config.backend)
            if not step_cls:
                raise ValueError(
                    f"Step references an unregistered backend '{config.backend}' "
                    f"of type '{ComponentEnum.STEP}'",
                )
            params = config.model_dump()
            params["app_context"] = self.app_context
            self.step_components.append(step_cls(**params))

    async def _close(self) -> None:
        self.step_components.clear()

    async def __call__(self, **kwargs) -> Response:
        context = RuntimeContext(**kwargs)
        for step_component in self.step_components:
            await step_component(context)
        return context.response
