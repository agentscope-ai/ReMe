"""Demo steps for smoke-testing the application stack."""

from ..base_step import BaseStep
from ...components.component_registry import R


@R.register("demo_echo")
class DemoEchoStep(BaseStep):
    """Echo back the incoming `query` field into `response.answer`."""

    async def execute(self):
        assert self.context is not None
        query = self.context.get("query", "")
        self.context.response.answer = f"echo: {query}"
        self.context.response.metadata["step"] = self.name
        return self.context.response


__all__ = ["DemoEchoStep"]
