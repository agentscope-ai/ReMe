"""Execute one due delegation in a fork of its source Main conversation."""

import asyncio

import frontmatter

from .base import BasePaAgentStep
from ....components import R
from ....schema.pa_task import PaTaskRun


@R.register("temp_agent_step")
class TempAgentStep(BasePaAgentStep):
    """Own one model attempt, its durable result and retry-safe submission."""

    async def execute(self):
        self._validate("task_id", "revision")
        identifier, revision = self.context.get("task_id"), self.context.get("revision")
        async with self.pa.temp_tasks.run_locks.setdefault(identifier, asyncio.Lock()):
            task = await asyncio.to_thread(self.pa.temp_tasks.get, identifier)
            if task is None or task.revision != revision or not task.runnable:
                raise asyncio.CancelledError()
            if task.run_at > self._now():
                raise ValueError("Temporary task is not due")
            if self.context.stop_event is not None and self.context.stop_event.is_set():
                raise asyncio.CancelledError()
            await self._run(task)
            saved = await asyncio.to_thread(self.pa.temp_tasks.get, identifier)
            self.context.response.answer = saved.model_dump(mode="json")
            self.context.response.success = saved.status == "completed"
            return self.context.response

    async def _run(self, task):
        tasks = self.pa.temp_tasks
        identifier, revision = task.id, task.revision
        recovered_late = (
            self.pa.started_at is not None and task.created_at < self.pa.started_at and task.run_at < self.pa.started_at
        )
        try:
            started_at = self._now()
            self._start_activity(task, started_at)
            task = await tasks.patch(
                task.id,
                task.revision,
                delayed=task.delayed or recovered_late or (self._now() - task.run_at).total_seconds() > 5,
            )
            if task.result_path is None:
                task = await tasks.recover_result(task)
            if task.result_path is not None:
                result = await asyncio.to_thread(self.pa.files.read, self.workspace_path / task.result_path)
                content = frontmatter.loads(result).content
            else:
                if task.queried:
                    raise RuntimeError("Task query was interrupted without a saved result; resume the task to retry")
                task = await tasks.patch(
                    task.id,
                    task.revision,
                    status="running",
                    queried=True,
                    started_at=started_at,
                    agent_session_id=None,
                )
                reply = await self.agent_wrapper.reply(
                    self.prompt_format(
                        "query_execute",
                        query=task.query,
                        now=self._now().isoformat(),
                        run_at=task.run_at.isoformat(),
                    ),
                    resume=task.source_session_id,
                    fork_session=True,
                    activity_source="task",
                    activity_started_at=started_at,
                    activity_task_id=f"temp_{task.id}",
                )
                if reply["status"] != "completed":
                    raise RuntimeError(reply.get("error") or f"Task agent {reply['status']}")
                content = reply["result"]
                task = await tasks.save_result(
                    task.id,
                    task.revision,
                    content,
                    agent_session_id=reply["session_id"],
                )
            await tasks.send(task.id, task.revision, content)
        except Exception as exc:
            self.logger.exception(f"Temporary task {task.id} failed")
            try:
                await tasks.patch(
                    identifier,
                    revision,
                    status="sending" if task.result_path and isinstance(exc, OSError) else "failed",
                    error=str(exc)[:1000],
                )
            except asyncio.CancelledError:
                pass
        finally:
            self.pa.event_stream.finish_task_activity(f"temp_{task.id}")

    def _start_activity(self, task, started_at):
        self.pa.event_stream.start_task_activity(
            PaTaskRun(
                task_id=f"temp_{task.id}",
                query=task.query,
                started_at=started_at,
            ),
        )
