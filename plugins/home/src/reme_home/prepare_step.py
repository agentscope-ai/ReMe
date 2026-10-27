"""Expose one-time preparation through the ordinary ReMe Job lifecycle."""

from reme.steps.base_step import BaseStep, Ref

from .preparation import HomePreparation


class HomePrepareStep(BaseStep):
    """Accept a workspace YAML file or a Python configuration dictionary."""

    preparation: HomePreparation = Ref(HomePreparation, "home.preparation")

    async def execute(self):
        """Prepare once and return the persisted completion information."""
        if self.context.get("_allowed_paths") is not None:
            raise ValueError("home_prepare requires unrestricted access to its workspace output directories")
        result = await self.preparation.prepare(
            spec=self.context.get("spec"),
            spec_path=self.context.get("spec_path"),
        )
        self.context.response.answer = result
        return self.context.response
