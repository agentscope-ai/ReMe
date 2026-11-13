"""Browse task results and record an explicit user-read receipt."""

from ..base import BasePaStep
from ....components import R


@R.register("task_results_step")
class TaskResultsStep(BasePaStep):
    """Keep transport parameters and permissions separate from file projections."""

    async def execute(self):
        self.require_paths(self.pa.tasks.root_path)
        self.context.response.answer = await self.pa.views.task_results(self.context.data)
        return self.context.response
