"""Thin Job adapters; operations are fixed by Step configuration, never model input."""

from reme.steps.base_step import BaseStep, Ref

from .component import HomeCode
from .models import ARGUMENTS


class HomeCodeStep(BaseStep):
    """Expose one configured candidate operation using the normal ReMe Job lifecycle."""

    code: HomeCode = Ref(HomeCode, "home.code")

    async def execute(self):
        """Leave repository location and operation selection under host control."""
        if self.context.get("_allowed_paths") is not None:
            raise ValueError("Home code Jobs require unrestricted access to their managed workspace")
        operation = self.kwargs["operation"]
        fields = ARGUMENTS[operation].model_fields
        arguments = {name: self.context.get(name) for name in fields if name in self.context}
        self.context.response.answer = await self.code.run(operation, **arguments)
        return self.context.response
