"""Execute due task queries through the existing task-agent Job."""

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R


@R.register("run_due_tasks_step")
class RunDueTasksStep(BaseStep):
    """Delegate due-slot checks and persistent execution records to the PA component."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        for path in (self.pa.tasks_path, self.pa.task_runs_path):
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to access scheduled task metadata")
        runner = self.get_job(self.kwargs.get("task_job", "task_agent"))
        if runner is None:
            raise RuntimeError("run_due_tasks requires a configured task_agent Job")
        self.context.response.answer = await self.pa.run_due_tasks(runner, stop_event=self.context.stop_event)
        return self.context.response
