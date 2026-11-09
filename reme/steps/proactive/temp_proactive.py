"""Lifecycle-owned preparation and execution of explicitly requested one-shot work."""

import asyncio
from datetime import timedelta

import frontmatter

from .base_pa_step import BasePaStep
from ...components import R
from ...schema.pa_task import PaTaskRun


@R.register("temp_proactive_step")
class TempProactiveStep(BasePaStep):
    """Create independent workers, cancel stale revisions, and close every worker."""

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("Temporary task execution requires an application-owned background job")
        workers = {}
        try:
            while not stop.is_set():
                records = await asyncio.to_thread(self.pa.temp_tasks.load)
                desired = {
                    task.id: task
                    for task in records
                    if task.enabled and task.status not in {"completed", "cancelled", "failed", "paused"}
                }
                for identifier, (revision, worker) in list(workers.items()):
                    task = desired.get(identifier)
                    if worker.done() or task is None or task.revision != revision:
                        if not worker.done():
                            worker.cancel()
                        await asyncio.gather(worker, return_exceptions=True)
                        del workers[identifier]
                for identifier, task in desired.items():
                    if identifier not in workers:
                        workers[identifier] = (task.revision, asyncio.create_task(self._run(task)))
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
        finally:
            for _, worker in workers.values():
                worker.cancel()
            await asyncio.gather(*(worker for _, worker in workers.values()), return_exceptions=True)
        return self.context.response

    async def _run(self, task):
        tasks = self.pa.temp_tasks
        identifier, revision = task.id, task.revision
        recovered_late = (
            self.pa.started_at is not None and task.created_at < self.pa.started_at and task.run_at < self.pa.started_at
        )
        try:
            self._start_activity(task)
            await self._ensure_memory()
            task = await self._prepare(task)
            self.pa.finish_task_activity(f"temp_{identifier}")
            task = await self._wait_until(task, task.run_at)
            self._start_activity(task)
            task = await tasks.patch(
                task.id,
                task.revision,
                delayed=task.delayed or recovered_late or (self._now() - task.run_at).total_seconds() > 5,
            )
            if task.result_path is None:
                task = await tasks.recover_text(task, "result.md")
            if task.result_path is not None:
                result = await asyncio.to_thread(self.pa.tasks.read, self.workspace_path / task.result_path)
                await tasks.send(task.id, task.revision, frontmatter.loads(result).content)
                return
            task = await tasks.patch(
                task.id,
                task.revision,
                status="running",
                started_at=task.started_at or self._now(),
            )
            summary = await asyncio.to_thread(self.pa.tasks.read, self.workspace_path / task.handoff_path)
            reply = await self.agent_wrapper.reply(
                self.prompt_format("query_execute", query=task.query),
                system_prompt=self._system_prompt(
                    now=self._now().isoformat(),
                    handoff=summary,
                    memory=await asyncio.to_thread(self.pa.tasks.read, self._memory_path()),
                    source_session_id=task.source_session_id,
                    source_path=await self._source_path(task),
                ),
                job_tools=list(self.kwargs.get("job_tools") or []),
                injected_job_kwargs={
                    "pa_caller": "temporary",
                    "temp_task_id": task.id,
                    "temp_revision": task.revision,
                },
                activity_source="task",
                activity_run_id=f"temp_{task.id}",
                activity_task_id=f"temp_{task.id}",
            )
            current = tasks.get(task.id)
            if current is not None and current.message_id:
                return
            self._completed(reply)
            raise RuntimeError("Temporary task finished without calling send_chat_message_immediate")
        except Exception as exc:
            self.logger.exception(f"Temporary task {task.id} failed")
            try:
                await tasks.patch(identifier, revision, status="failed", error=str(exc)[:1000])
            except asyncio.CancelledError:
                pass
        finally:
            self.pa.finish_task_activity(f"temp_{task.id}")

    def _start_activity(self, task):
        self.pa.start_task_activity(
            PaTaskRun(
                run_id=f"temp_{task.id}",
                task_id=f"temp_{task.id}",
                query=task.query,
                started_at=self._now(),
            ),
        )

    async def _wait_until(self, task, deadline):
        identifier, revision = task.id, task.revision
        while deadline > self._now():
            await asyncio.sleep(min(1, (deadline - self._now()).total_seconds()))
            task = await asyncio.to_thread(self.pa.temp_tasks.get, identifier)
            if (
                task is None
                or task.revision != revision
                or not task.enabled
                or task.status in {"cancelled", "completed", "paused", "failed"}
            ):
                raise asyncio.CancelledError()
        return task

    async def _prepare(self, task):
        tasks = self.pa.temp_tasks
        if task.handoff_path is None:
            task = await tasks.recover_text(task, "handoff.md")
        while task.handoff_path is None:
            if task.retry_at is not None:
                task = await self._wait_until(task, task.retry_at)
            self._start_activity(task)
            task = await tasks.patch(task.id, task.revision, status="preparing", error=None, retry_at=None)
            try:
                reply = await self.agent_wrapper.reply(
                    self._handoff_prompt("query_prepare", query=task.query, now=self._now().isoformat()),
                    resume=task.source_session_id,
                    fork_session=True,
                    activity_source="task",
                    activity_run_id=f"temp_{task.id}",
                    activity_task_id=f"temp_{task.id}",
                )
                self._completed(reply)
                if not reply["result"].strip():
                    raise RuntimeError("Context preparation returned an empty handoff")
            except Exception as exc:
                task = await tasks.patch(
                    task.id,
                    task.revision,
                    status="preparing",
                    error=str(exc)[:1000],
                    retry_at=self._now() + timedelta(seconds=30),
                )
                self.pa.finish_task_activity(f"temp_{task.id}")
                self.logger.warning(f"Temporary context preparation will retry: {exc}")
                continue
            task = await tasks.save_text(
                task.id,
                task.revision,
                "handoff.md",
                reply["result"],
                status="scheduled",
                error=None,
                retry_at=None,
            )
        return task

    async def _source_path(self, task):
        try:
            path = await self.agent_wrapper.app_server.get_session_path(task.source_session_id)
            if path is not None and await asyncio.to_thread(path.is_file):
                return str(path)
        except Exception as exc:
            self.logger.warning(f"Source transcript is unavailable: {exc}")
        return self.get_prompt("source_unavailable")

    @staticmethod
    def _completed(reply):
        if reply["status"] != "completed":
            raise RuntimeError(reply.get("error") or f"Temporary agent {reply['status']}")
