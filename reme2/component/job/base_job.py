"""Base job component for sequential step execution."""

from ..base_component import BaseComponent
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ComponentEnum
from ...schema import Response, ComponentConfig


@R.register("base")
class BaseJob(BaseComponent):
    """Base job that executes a sequence of steps.

    A job orchestrates multiple steps in sequence, passing a runtime context
    through each step. Steps are configured via ComponentConfig and instantiated
    lazily when the job starts.
    """

    component_type = ComponentEnum.JOB

    def __init__(
        self,
        name: str = "",
        description: str = "",
        parameters: dict | None = None,
        steps: list[ComponentConfig] | None = None,
        **kwargs,
    ):
        """Initialize the job.

        Args:
            name: Job name identifier.
            description: Human-readable description.
            parameters: Default parameters passed to steps.
            steps: List of step configurations to execute.
            **kwargs: Additional arguments passed to BaseComponent.
        """
        super().__init__(**kwargs)

        self.name: str = name
        self.description: str = description
        self.parameters: dict = parameters or {}
        self.step_configs: list[ComponentConfig] = steps or []
        self.steps: list = []

    async def _start(self) -> None:
        """Instantiate all configured steps."""
        assert self.app_context is not None, "app_context must be provided"
        for step_config in self.step_configs:
            if not step_config.backend:
                raise ValueError(f"{step_config.backend} backend is not specified.")

            backend_cls = R.get(ComponentEnum.STEP, step_config.backend)
            if not backend_cls:
                raise ValueError(f"{step_config.backend} is not registered.")

            step = backend_cls(
                language=self.app_context.app_config.language,
                **step_config.model_dump(exclude={"backend"}),
            )
            self.steps.append(step)

    async def _close(self) -> None:
        """Clear all instantiated steps."""
        self.steps.clear()

    async def __call__(self, **kwargs) -> Response:
        """Execute all steps sequentially.

        Args:
            **kwargs: Parameters passed to the runtime context.

        Returns:
            The final response from the runtime context.
        """
        context = RuntimeContext(**kwargs)
        for step in self.steps:
            await step(context)

        return context.response
