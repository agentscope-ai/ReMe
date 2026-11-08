"""Reconcile user-edited definitions into lifecycle-owned ReMe CronJobs."""

import asyncio

from ..base_step import BaseStep, Ref
from ...components import PaRuntime, R
from ...components.job import CronJob
from ...utils.async_utils import finish_cleanup


@R.register("task_schedule_step")
class TaskScheduleStep(BaseStep):
    """Own independent task schedules until the background job stops."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("task_schedule_step requires an asyncio stop_event")
        template = self.get_job("task_agent")
        if template is None:
            raise RuntimeError("task_schedule_step requires the task_agent Job")
        template.check_enabled()
        jobs = {}
        signature = None

        async def close_jobs():
            results = await asyncio.gather(*(job.close() for job in jobs.values()), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result

        try:
            while not stop.is_set():
                try:
                    stat = self.pa.tasks_path.stat()
                    current = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
                    if current != signature:
                        definitions = await asyncio.to_thread(self.pa.tasks.load)
                        desired = {task.id: task for task in definitions if task.enabled}
                        for task_id in list(jobs):
                            job = jobs[task_id]
                            task = desired.get(task_id)
                            if task is None or job.kwargs["task"] != task.model_dump(mode="json"):
                                await job.close()
                                del jobs[task_id]
                        for task_id, task in desired.items():
                            if task_id in jobs:
                                continue
                            job = CronJob(
                                name=f"task_{task_id}",
                                app_context=self.app_context,
                                cron=task.cron,
                                steps=template.step_configs,
                                track_execution=True,
                                **{**template.kwargs, "task": task.model_dump(mode="json")},
                            )
                            await job.start()
                            jobs[task_id] = job
                        signature = current
                except (OSError, ValueError) as exc:
                    self.logger.error(f"Cannot update task schedules: {exc}")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
        finally:
            await finish_cleanup(close_jobs())
        return self.context.response
