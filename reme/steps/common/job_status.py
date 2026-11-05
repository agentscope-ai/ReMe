"""Query in-memory execution and step status."""

from ..base_step import BaseStep
from ...components import R


@R.register("job_status_step")
class JobStatusStep(BaseStep):
    """Return recent executions grouped by job name."""

    async def execute(self):
        assert self.context is not None
        assert self.app_context is not None
        self.context.response.answer = self.app_context.get_job_status(
            name=self.context.get("name"),
            status=self.context.get("status"),
            limit=self.context.get("limit", 20),
        )
        return self.context.response
