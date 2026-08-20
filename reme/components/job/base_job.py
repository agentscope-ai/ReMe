"""Base job component for sequential step execution."""

import time
from typing import TYPE_CHECKING

from ..base_component import BaseComponent
from ..component_registry import R
from ..runtime_context import RuntimeContext
from ...enumeration import ComponentEnum
from ...schema import ComponentConfig, Response
from ...utils import global_counter_inc

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

    def _record_call(self) -> None:
        """Increment this job's application-lifetime call counter and mark it running.

        ``__job_last_run`` feeds the idle gate (``wait_for_idle_step``): metadata
        is process-local, so a restart always leaves every job looking idle.
        """
        metadata = getattr(self.app_context, "metadata", None)
        if isinstance(metadata, dict):
            global_counter_inc(metadata, ["__job_counter", self.name])
            last_run = metadata.setdefault("__job_last_run", {})
            entry = dict(last_run.get(self.name) or {})
            entry.update(running=True, last_start=time.monotonic())
            last_run[self.name] = entry

    def _finish_call(self) -> None:
        """Mark this job's run as finished for the idle gate."""
        metadata = getattr(self.app_context, "metadata", None)
        if isinstance(metadata, dict):
            last_run = metadata.setdefault("__job_last_run", {})
            entry = dict(last_run.get(self.name) or {})
            entry.update(running=False, last_end=time.monotonic())
            last_run[self.name] = entry

    async def __call__(self, **kwargs) -> Response:
        """Run all steps in order, capturing any failure into the response."""
        self._record_call()
        merged = {**self.kwargs, **kwargs}
        context = RuntimeContext(**merged)
        try:
            for step in self._build_steps():
                await step(context)
        except Exception as e:
            self.logger.exception(f"Failed to execute job: {e}")
            context.response.success = False
            context.response.answer = str(e)
        finally:
            self._finish_call()
        return context.response
