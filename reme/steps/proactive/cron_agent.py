"""Execute, publish optional cumulative findings, then save a recoverable handoff."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import frontmatter

from .base_pa_step import BasePaStep
from ..file_io._path import _check_path_permission
from ...components import R
from ...schema.pa_task import PaTask, PaTaskRun, PaTaskRecovery
from ...schema.token_usage import TokenUsage
from ...schema.connector_result import ConnectorResult


@R.register("cron_agent_step")
class CronAgentStep(BasePaStep):
    """Execute, summarize and prepare the next handoff in one session."""

    async def execute(self):
        self._validate("task", "_allowed_paths")
        task = PaTask.model_validate(self.context.get("task"))
        if not task.enabled:
            self.context.response.answer = {"task_id": task.id, "status": "skipped", "reason": "disabled"}
            return self.context.response
        lock = self.pa.tasks.run_locks.setdefault(task.id, asyncio.Lock())
        async with lock:
            if task.message is not None:
                return await self._fixed_message(task)
            return await self._run_task(task)

    async def _send_immediate(self, task, run, content):
        try:
            definitions = await asyncio.to_thread(self.pa.tasks.load)
        except (OSError, ValueError):
            definitions = [PaTask.model_validate(value) for value in self.pa.tasks.scheduled.values()]
        if task not in definitions or not task.enabled:
            raise RuntimeError("Immediate task changed or was cancelled before publication")
        result = await self.pa.reminders.submit_immediate(
            f"cron_{run.run_id}",
            content,
            session_id=task.source_session_id,
            task_id=task.id,
            task_definition=task.model_dump(mode="json"),
        )
        if result["status"] == "discarded":
            raise RuntimeError("Task changed before immediate presentation")
        run.message_id = result["id"]

    async def _fixed_message(self, task):
        for path in (self.pa.chat_inbox_path, self.pa.cron_runs_path, self.pa.cron_path):
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to deliver scheduled messages")
        run = PaTaskRun(task_id=task.id, query=task.message, started_at=self._now())
        try:
            await self.pa.tasks.record(run)
            await self._send_immediate(task, run, task.message)
            run.status = "succeeded"
        except BaseException as exc:
            run.status, run.error = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed", str(exc)[
                :1000
            ]
            raise
        finally:
            run.finished_at = self._now()
            try:
                await self.pa.tasks.record(run)
            finally:
                self.pa.finish_task_activity(run.run_id)
        self.context.response.answer = run.model_dump(mode="json")
        return self.context.response

    async def _run_task(self, task):
        if task.delivery != "immediate":
            config = await self.pa.get_proactive_config()
            if config.tasks_follow_active_hours and not config.is_active(self._now()):
                self.context.response.answer = {
                    "task_id": task.id,
                    "status": "skipped",
                    "reason": "outside_active_hours",
                }
                return self.context.response
        files = self.pa.tasks
        paths = [self.pa.cron_runs_path, files.task_path(task.id)]
        if task.delivery == "immediate":
            paths.extend([self.pa.chat_inbox_path, self.pa.cron_path])
        for path in paths:
            if not _check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to execute scheduled tasks")
        saved = await asyncio.to_thread(files.recovery, task)
        run = saved.run if saved is not None else PaTaskRun(task_id=task.id, query=task.query, started_at=self._now())
        run.status, run.finished_at, run.error = "running", None, None
        options = {
            "system_prompt": self._system_prompt(
                now=self._now().isoformat(),
                memory=await asyncio.to_thread(files.read, await self._ensure_memory()),
                handoff=(await asyncio.to_thread(files.handoff, task.id)).content or self.get_prompt("no_previous"),
            ),
            "job_tools": list(self.kwargs.get("job_tools") or []),
            "activity_source": "task",
            "activity_run_id": run.run_id,
            "activity_task_id": task.id,
            "immediate": task.delivery == "immediate",
        }
        try:
            await files.record(run)
            if saved is None:
                prefetched = await self._preflight(task)
                if prefetched is False:
                    run.status, run.reason = "skipped", "screen_unchanged"
                    self.context.response.answer = {"task_id": task.id, "status": run.status, "reason": run.reason}
                    return self.context.response
                self._bind_preflight(options, prefetched)
                reply = await self._call(task.query, run, options)
                if reply is None:
                    return self.context.response
                now = self._now()
                target = files.task_path(task.id) / now.date().isoformat() / f"{now:%H%M%S}.md"
                if target.exists():
                    target = target.with_stem(f"{target.stem}-{run.run_id}")
                run.phase = "summary"
                saved = PaTaskRecovery(
                    task=task,
                    run=run,
                    target_path=self.to_workspace_relative(files.checked(target)),
                    prefetched=prefetched,
                    query_result=reply["result"],
                    usage=TokenUsage.model_validate(reply.get("usage") or {}),
                )
                await files.checkpoint(saved)
            self._bind_preflight(options, saved.prefetched)
            if run.phase == "summary" and not await self._summarize(saved, options):
                return self.context.response
            if task.delivery == "immediate":
                if not _check_path_permission(
                    self.workspace_path,
                    self.pa.chat_inbox_path,
                    self.context.get("_allowed_paths"),
                ):
                    raise PermissionError("no permission to deliver scheduled messages")
                content = frontmatter.loads(files.read(self.workspace_path / run.result_path)).content
                await self._send_immediate(task, run, content)
                await files.checkpoint(saved)
            if not await self._handoff(saved, options):
                return self.context.response
            run.status = "succeeded"
            run.handoff_path = self.to_workspace_relative(files.handoff_path(task.id))
            await files.clear_recovery(saved)
            self.context.response.answer = {
                "task_id": task.id,
                "status": run.status,
                "path": run.result_path,
                "handoff_path": run.handoff_path,
                "reason": run.reason,
            }
            self.context.response.metadata.update(
                session_id=run.agent_session_id,
                status="completed",
                usage=saved.usage,
            )
        except asyncio.CancelledError:
            run.status, run.error = "interrupted", "Execution cancelled; completed phases are retained for recovery"
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

    async def _preflight(self, task):
        if task.preflight != "screen_capture":
            return None
        result = await self.run_job(
            "screen_capture",
            stop_event=self.context.stop_event,
            _allowed_paths=self.context.get("_allowed_paths"),
            _screen_baseline=await asyncio.to_thread(self.pa.tasks.screen_baseline, task.id),
            _screen_scope=f"task:{task.id}",
        )
        if not result.success:
            raise RuntimeError(f"Screen preflight failed: {result.answer}")
        images = ConnectorResult.model_validate(result.answer)
        return {"result": images.model_dump(), "frames": result.metadata["screen_capture"]} if images.paths else False

    @staticmethod
    def _bind_preflight(options, prefetched):
        if prefetched is not None:
            options["injected_job_kwargs"] = {"_screen_preflight": prefetched}

    async def _call(self, query, run, options):
        if self.context.stop_event is not None and self.context.stop_event.is_set():
            raise asyncio.CancelledError()
        if run.agent_session_id is not None:
            options = {**options, "resume": run.agent_session_id}
        reply = await self._reply(query, task=True, **options)
        if reply is None:
            run.status, run.error = "interrupted", "outside_active_hours"
            self.context.response.answer = {"task_id": run.task_id, "status": run.status, "reason": run.error}
            return None
        run.agent_session_id = reply["session_id"]
        if reply["status"] != "completed":
            raise RuntimeError(reply.get("error") or f"Task agent {reply['status']}")
        return reply

    async def _summarize(self, saved, options):
        files, run = self.pa.tasks, saved.run
        target = files.checked(self.workspace_path / saved.target_path)
        if target.exists():
            # Recover a crash after exclusive publication but before the phase checkpoint.
            post = frontmatter.loads(files.read(target))
            if post.get("run_id") != run.run_id or post.get("task_id") != saved.task.id or not post.content.strip():
                raise FileExistsError(f"Task snapshot already exists: {target}")
            run.result_path = saved.target_path
        else:
            with TemporaryDirectory(prefix=".reme-task-", dir=self.workspace_path) as temporary:
                staging = Path(temporary) / "summary.md"
                previous = await asyncio.to_thread(files.paths, saved.task.id)
                reply = await self._call(
                    self.prompt_format(
                        "query_summary_immediate" if saved.task.delivery == "immediate" else "query_summary",
                        previous_paths="\n".join(str(path) for path in previous[:2]) or self.get_prompt("no_previous"),
                        staging_path=staging,
                        now=self._now().isoformat(),
                    ),
                    run,
                    options,
                )
                if reply is None:
                    return False
                self._add_usage(saved, reply)
                if reply["result"].strip() == self.get_prompt("no_update"):
                    if saved.task.delivery == "immediate":
                        raise ValueError("An explicitly scheduled message cannot be skipped as NO_UPDATE")
                    if staging.exists():
                        raise ValueError("NO_UPDATE must not accompany a summary file")
                    run.reason = "no_new_information"
                else:
                    await files.publish(
                        staging,
                        target,
                        saved.task.id,
                        run.run_id,
                        screen_capture=saved.prefetched["frames"] if saved.prefetched is not None else None,
                    )
                    run.result_path = saved.target_path
        run.phase = "handoff"
        await files.checkpoint(saved)
        return True

    async def _handoff(self, saved, options):
        files = self.pa.tasks
        previous = await asyncio.to_thread(files.handoff, saved.task.id)
        if previous.get("run_id") == saved.run.run_id:
            return True
        reply = await self._call(
            self._handoff_prompt(
                "query_handoff",
                query=saved.task.query,
                result_path=saved.run.result_path or self.get_prompt("no_result"),
                now=self._now().isoformat(),
            ),
            saved.run,
            options,
        )
        if reply is None:
            return False
        self._add_usage(saved, reply)
        await files.save_handoff(saved, reply["result"])
        return True

    @staticmethod
    def _add_usage(saved, reply):
        saved.usage = TokenUsage.combine([saved.usage, TokenUsage.model_validate(reply.get("usage") or {})])
