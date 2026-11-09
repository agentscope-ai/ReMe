"""Run each recurring task in one persistent native conversation."""

import asyncio

from .base import BasePaAgentStep
from ...file_io._path import _check_path_permission
from ....components import R
from ....components.pa.reminders import ReminderHistoryChanged
from ....schema.pa_task import PaTask, PaTaskRun
from ....schema.token_usage import TokenUsage


@R.register("cron_agent_step")
class CronAgentStep(BasePaAgentStep):
    """Execute, consolidate and optionally announce a task in one timed query."""

    async def execute(self):
        self._validate("task", "_allowed_paths")
        task = PaTask.model_validate(self.context.get("task"))
        if not task.enabled:
            self.context.response.answer = {"task_id": task.id, "status": "skipped", "reason": "disabled"}
            return self.context.response
        files = self.pa.tasks
        for path in (self.pa.cron_runs_path, files.task_path(task.id)):
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to execute scheduled tasks")
        async with files.run_locks.setdefault(task.id, asyncio.Lock()):
            return await self._run(task)

    async def _run(self, task):
        files = self.pa.tasks
        state = await asyncio.to_thread(files.state, task)
        config = await self.pa.get_proactive_config()
        if state.completion is None and config.tasks_follow_active_hours and not config.is_active(self._now()):
            self.context.response.answer = {"task_id": task.id, "status": "skipped", "reason": "outside_active_hours"}
            return self.context.response
        run = state.run
        if (
            run is None
            or run.status in {"succeeded", "skipped"}
            or (state.completion is None and (run.status == "failed" or run.queried))
        ):
            run = PaTaskRun(
                task_id=task.id,
                query=task.query,
                started_at=await asyncio.to_thread(files.next_start, task.id),
                target_session_id=self.pa.session_id,
            )
            state.run = run
            state.completion = None
            state.usage = TokenUsage()
        run.status, run.error, run.reason, run.finished_at = "running", None, None, None
        with files.execution_scope(state):
            try:
                await files.record(run)
                if self.context.stop_event is not None and self.context.stop_event.is_set():
                    raise asyncio.CancelledError()
                if state.completion is not None:
                    try:
                        await files.finish(state)
                    except ReminderHistoryChanged:
                        state.completion.reminder = ""
                        run.reason = "stale_reminder"
                        await files.finish(state)
                else:
                    result = await self._within_active_hours(lambda: self._query(state), task=True)
                    if result is None and run.status != "succeeded":
                        run.status, run.error, run.reason = (
                            "interrupted",
                            "outside_active_hours",
                            "outside_active_hours",
                        )
            except asyncio.CancelledError:
                if run.status != "succeeded":
                    run.status, run.error = "interrupted", "Execution cancelled"
                raise
            except Exception as exc:
                if run.status != "succeeded":
                    run.status, run.error = "failed", str(exc)[:1000]
                    raise
            finally:
                run.finished_at = run.finished_at or self._now()
                try:
                    await files.save(state)
                    await files.record(run)
                finally:
                    self.pa.event_stream.finish_task_activity(task.id)
        self.context.response.answer = files.answer(state)
        self.context.response.metadata.update(
            session_id=state.session_id,
            status=run.status,
            usage=state.usage.model_dump(),
        )
        return self.context.response

    async def _query(self, state):
        files, run = self.pa.tasks, state.run
        await self._ensure_memory()
        options = self._agent_options(
            injected={
                "pa_caller": "cron",
                "cron_task_id": run.task_id,
                "_allowed_paths": self.context.get("_allowed_paths"),
            },
        )
        if not await self._session_exists(state.session_id):
            state.session_id = await self.agent_wrapper.create_session(**options)
        run.agent_session_id = state.session_id
        staging = files.staging_path(run)
        await asyncio.to_thread(staging.parent.mkdir, parents=True, exist_ok=True)
        warnings = []
        previous = await asyncio.to_thread(lambda: next(files.results(state.task.id, warnings=warnings), None))
        if warnings:
            self.context.response.metadata["task_result_warnings"] = warnings[:20]
        query = self.prompt_format(
            "query",
            now=self._now().isoformat(),
            query=state.task.query,
            memory_path=self._memory_path(),
            previous_path=str(previous[0]) if previous else self.get_prompt("no_previous"),
            staging_path=staging,
        )
        run.queried = True
        await files.save(state)
        reply = await self._within_active_hours(
            lambda: self.agent_wrapper.reply(
                query,
                resume=state.session_id,
                activity_source="task",
                activity_started_at=run.started_at,
                activity_task_id=run.task_id,
                **options,
            ),
            task=True,
        )
        if reply is None:
            if run.status != "succeeded":
                run.status, run.error, run.reason = "interrupted", "outside_active_hours", "outside_active_hours"
            return True
        state.usage = TokenUsage.model_validate(reply.get("usage") or {})
        if run.status == "succeeded":
            return True
        if reply.get("status") != "completed":
            raise RuntimeError(reply.get("error") or f"Cron task {reply.get('status')}")
        if reply["result"].strip() == self.get_prompt("no_update"):
            if staging.exists():
                raise ValueError("NO_UPDATE must not accompany a summary document")
        elif not staging.exists():
            raise ValueError("Cron query must write its summary document or return NO_UPDATE")
        await files.complete(state.task.id)
        return True
