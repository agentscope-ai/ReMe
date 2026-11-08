"""Execute a task and publish its cumulative summary after two agent queries."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from .base_pa_step import BasePaStep
from ..file_io._path import _check_path_permission
from ...components import R
from ...schema.pa_task import PaTask, PaTaskRun
from ...schema.token_usage import TokenUsage
from ...schema.connector_result import ConnectorResult


@R.register("task_agent_step")
class TaskAgentStep(BasePaStep):
    """Collect evidence, then continue the same session to update a file-native summary."""

    async def execute(self):
        self._validate("task", "_allowed_paths")
        task = PaTask.model_validate(self.context.get("task"))
        lock = self.pa.tasks.run_locks.setdefault(task.id, asyncio.Lock())
        async with lock:
            return await self._run_task(task)

    async def _run_task(self, task):
        config = await self.pa.get_proactive_config()
        if config.tasks_follow_active_hours and not config.is_active(self._now()):
            self.context.response.answer = {"task_id": task.id, "status": "skipped", "reason": "outside_active_hours"}
            return self.context.response
        files = self.pa.tasks
        for path in (self.pa.task_runs_path, files.daily_path):
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to execute scheduled tasks")
        run = PaTaskRun(task_id=task.id, query=task.query, started_at=self._now())
        try:
            await files.record(run)
            capture, prefetched = None, None
            if task.preflight == "screen_capture":
                baseline = await asyncio.to_thread(files.screen_baseline, task.id)
                result = await self.run_job(
                    "screen_capture",
                    stop_event=self.context.stop_event,
                    _allowed_paths=self.context.get("_allowed_paths"),
                    _screen_baseline=baseline,
                    _screen_scope=f"task:{task.id}",
                )
                if not result.success:
                    raise RuntimeError(f"Screen preflight failed: {result.answer}")
                images = ConnectorResult.model_validate(result.answer)
                if not images.paths:
                    run.status, run.reason = "skipped", "screen_unchanged"
                    self.context.response.answer = {"task_id": task.id, "status": run.status, "reason": run.reason}
                    return self.context.response
                capture = result.metadata["screen_capture"]
                prefetched = {"result": images.model_dump(), "frames": capture}
            await self._ensure_memory()
            options = {
                "system_prompt": self._system_prompt(now=self._now().isoformat()),
                "job_tools": list(self.kwargs.get("job_tools") or []),
                "activity_source": "task",
                "activity_run_id": run.run_id,
                "activity_task_id": task.id,
            }
            if prefetched is not None:
                options["injected_job_kwargs"] = {"_screen_preflight": prefetched}
            if self.context.stop_event is not None and self.context.stop_event.is_set():
                raise asyncio.CancelledError()
            reply = await self._reply(task.query, task=True, **options)
            if reply is None:
                return self._skip_run(run)
            run.agent_session_id = reply["session_id"]
            self._check_reply(reply)
            usages = [reply.get("usage")]
            if self.context.stop_event is not None and self.context.stop_event.is_set():
                raise asyncio.CancelledError()
            now = self._now()
            target = files.checked(files.daily_path / now.date().isoformat() / "cron" / task.id / f"{now:%H%M%S}.md")
            if target.exists():
                raise FileExistsError(f"Task snapshot already exists: {target}")
            with TemporaryDirectory(prefix=".reme-task-", dir=self.workspace_path) as temporary:
                previous = await asyncio.to_thread(files.paths, task.id)
                query = self.prompt_format(
                    "query_summary",
                    previous_paths="\n".join(str(path) for path in previous[:2]) or self.get_prompt("no_previous"),
                    staging_path=Path(temporary) / "summary.md",
                    result_path=target,
                    now=now.isoformat(),
                )
                reply = await self._reply(query, task=True, resume=run.agent_session_id, **options)
                if reply is None:
                    return self._skip_run(run)
                self._check_reply(reply)
                usages.append(reply.get("usage"))
                usage = TokenUsage.combine([TokenUsage.model_validate(usage) for usage in usages if usage is not None])
                await files.publish(
                    Path(temporary) / "summary.md",
                    target,
                    task.id,
                    run.run_id,
                    screen_capture=capture,
                )
                run.result_path = self.to_workspace_relative(target)
                run.status = "succeeded"
                self.context.response.answer = {"task_id": task.id, "path": run.result_path}
                self.context.response.metadata.update(
                    session_id=run.agent_session_id,
                    status=reply["status"],
                    usage=usage,
                )
        except asyncio.CancelledError:
            run.status, run.error = (
                "interrupted",
                "Execution cancelled; inspect snapshots for any completed publication",
            )
            raise
        except Exception as exc:
            run.status, run.error = "failed", str(exc)[:1000]
            raise
        finally:
            run.finished_at = self._now()
            try:
                await files.record(run)
            finally:
                self.pa.finish_task_activity(run.run_id)
        return self.context.response

    def _skip_run(self, run):
        run.status, run.error = "interrupted", "outside_active_hours"
        self.context.response.answer = {"task_id": run.task_id, "status": run.status, "reason": run.error}
        return self.context.response

    @staticmethod
    def _check_reply(reply):
        if reply["status"] != "completed":
            raise RuntimeError(reply.get("error") or f"Task agent {reply['status']}")
