"""Schedule one-shot tasks and supervise revision-bound execution jobs."""

import asyncio

from ._lifecycle import cancel_workers, require_stop_event, wait_for_stop
from ..base import BasePaStep
from ....components import R


@R.register("temp_task_loop_step")
class TempTaskLoopStep(BasePaStep):
    """Start due tasks and await every cancellation before shutdown."""

    async def execute(self):
        stop = require_stop_event(self.context.stop_event)
        job = self.get_job("temp_agent")
        if job is None:
            raise RuntimeError("Temporary task supervision requires the temp_agent Job")
        job.check_enabled()
        workers = {}
        try:
            while not stop.is_set():
                records = await asyncio.to_thread(self.pa.temp_tasks.load)
                desired = {task.id: task for task in records if task.runnable}
                for identifier, (revision, worker) in list(workers.items()):
                    task = desired.get(identifier)
                    if worker.done() or task is None or task.revision != revision:
                        await cancel_workers([worker])
                        del workers[identifier]
                for identifier, task in desired.items():
                    if identifier not in workers and task.run_at <= self.pa.now():
                        worker = asyncio.create_task(
                            self.run_job("temp_agent", task_id=task.id, revision=task.revision, stop_event=stop),
                        )
                        workers[identifier] = (task.revision, worker)
                await wait_for_stop(stop, 0.5)
        finally:
            await cancel_workers([worker for _, worker in workers.values()])
        return self.context.response
