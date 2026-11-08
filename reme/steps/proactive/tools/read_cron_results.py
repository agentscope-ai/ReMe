"""Consume cumulative cron results with durable recovery context."""

from ...base_step import BaseStep, Ref
from ...file_io._path import _check_path_permission
from ....components import PaRuntime, R


@R.register("read_cron_results_step")
class ReadCronResultsStep(BaseStep):
    """Read up to three unread snapshots per configured task, including previous days."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        round_id = self.context.get("pa_round_id")
        if self.pa.current_round is None or self.pa.current_round.id != round_id:
            raise RuntimeError("read_cron_results requires a live proactive round")
        for path in (self.pa.cron_path, self.pa.tasks.root_path, self.pa.proactive_context_path):
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to consume task results")
        self.context.response.answer = await self.pa.tasks.consume(self.context.get("limit", 3), round_id)
        return self.context.response
