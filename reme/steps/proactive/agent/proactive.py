"""Consume one event batch in a fresh session and persist a durable completion receipt."""

import asyncio
from datetime import timedelta
import math
from time import monotonic

import frontmatter

from .base import BaseAgentStep
from ....components import R
from ....schema.proactive_agent import ProactiveRunResult
from ....schema import collect_reply
from ....utils.async_utils import complete_in_thread


@R.register("proactive_agent_step")
class ProactiveAgentStep(BaseAgentStep):
    """Restore context, process inputs and publish validated reminder documents."""

    def __init__(self, timeout_seconds=1200, **kwargs):
        super().__init__(**kwargs)
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = timeout_seconds

    def _paths(self, agent, now):
        runtime = self.agent_runtime
        daily = runtime.files.path(self.config_value("daily_dir"))
        agent_dir = runtime.agents.directory_for(agent)
        reminder_dir = agent_dir / "reminder"
        day = now.strftime("%Y-%m-%d")
        summary = runtime.files.checked(daily / day / f"{agent.agent_name}.md")
        yesterday = (now.date() - timedelta(days=1)).isoformat()
        yesterday_summary = runtime.files.checked(daily / yesterday / f"{agent.agent_name}.md")
        stem = f"{now:%H%M%S}"
        collision = 0
        while True:
            filename = f"{stem}{'_' + str(collision) if collision else ''}.md"
            detail = runtime.files.checked(daily / day / f"{agent.agent_name}_{filename}")
            reminder = runtime.files.checked(reminder_dir / day / filename)
            if not detail.exists() and not reminder.exists():
                break
            collision += 1
        return {
            "daily_dir": daily,
            "reminder_dir": reminder_dir,
            "task_path": runtime.agents.task_path(agent),
            "today_summary_path": summary,
            "yesterday_summary_path": yesterday_summary,
            "record_path": detail,
            "reminder_path": reminder,
        }

    def session_options(self, agent, paths, now):
        """Build the full environment for one automatic run."""
        options = {
            "config": {"skills.include_instructions": False},
            "job_tools": self.kwargs.get("job_tools") or [],
            "system_prompt": self.prompt_format(
                "system_prompt",
                workspace_dir=self.workspace_path,
                agent_name=agent.agent_name,
                app_timezone=self.app_context.app_config.timezone or str(now.tzinfo),
                sleep_reminder_rule=self.get_prompt("sleep_reminder_rule"),
                **{key: str(value) for key, value in paths.items()},
                now=now.isoformat(),
                proactive=agent.agent_type == "proactive",
                cron=agent.agent_type == "cron",
            ),
            "injected_job_kwargs": {"_agent_name": agent.agent_name},
            "approval_mode": "deny_all",
        }
        if (sandbox := self.kwargs.get("sandbox", "workspace-write")) is not None:
            options["sandbox"] = sandbox
        return options

    def _record_query(self, agent, events):
        cron = agent.agent_type == "cron"
        files = self.agent_runtime.files
        inputs = []
        if not cron:
            for number, event in enumerate(events, 1):
                inputs.append(
                    self.prompt_format(
                        "event_input",
                        number=number,
                        event_ts=event["event_ts"],
                        source=event["source"],
                        description=event["description"],
                        paths="\n".join(str(files.path(path)) for path in event["paths"]),
                        has_files=bool(event["paths"]),
                        no_files=not event["paths"],
                    ),
                )
        return self.prompt_format(
            "query_record",
            input_content="\n\n".join(inputs),
            proactive=not cron,
            cron=cron,
        )

    def _read_optional(self, path):
        """Only a missing optional file is treated as absent."""
        try:
            return self.agent_runtime.files.read(path)
        except FileNotFoundError:
            return None

    def _read_document(self, path, *, action=None, allow_empty_skip=False):
        """Require an explicit decision and validate the body."""
        document = frontmatter.loads(self.agent_runtime.files.read(path))
        if document.get("action") not in ("write", "skip"):
            raise ValueError(f"{path}: action must be write or skip")
        if not document.content.strip() and not (allow_empty_skip and document["action"] == "skip"):
            raise ValueError(f"{path}: document body must not be blank")
        if action is not None and document["action"] != action:
            raise ValueError(f"{path}: action must remain {action}")
        return document

    def _reminder_result(self, paths):
        """Read business output from documents, never from the model's reply text."""
        files = self.agent_runtime.files
        document = self._read_document(paths["reminder_path"])
        if document["action"] == "skip":
            return ProactiveRunResult(notifications=[])
        self._read_document(paths["record_path"], action="write")
        return ProactiveRunResult(
            notifications=[
                {
                    "priority": document.get("priority"),
                    "message": document.content,
                    "paths": [
                        path.relative_to(files.workspace).as_posix()
                        for path in (paths["reminder_path"], paths["record_path"])
                    ],
                },
            ],
        )

    def _prepare_reminder(self, paths):
        """Validate the draft and add framework-owned related paths before summarizing."""
        result = self._reminder_result(paths)
        if result.notifications:
            notification = result.notifications[0]
            document = frontmatter.Post(
                notification.message,
                action="write",
                priority=notification.priority,
                paths=notification.paths,
            )
            self.agent_runtime.files.write(paths["reminder_path"], frontmatter.dumps(document) + "\n")
        return result

    def _read_artifacts(self, paths, summary_before, record, reminder):
        """Require a summary of new progress and preserve already validated documents."""
        files = self.agent_runtime.files
        if files.read(paths["record_path"]) != record or files.read(paths["reminder_path"]) != reminder:
            raise ValueError("summary stage must not change the record or reminder")
        summary = self._read_optional(paths["today_summary_path"])
        if not summary or not summary.strip():
            raise ValueError("new progress must have a nonempty daily summary")
        if summary == summary_before:
            raise ValueError("new progress must update its daily summary")
        return self._reminder_result(paths)

    async def _run_stage(self, agent_name, session_id, stage, query, path, check, options):
        """Observe one model turn, validate its file, and log the actual content change."""
        started = monotonic()
        label = f"Proactive agent={agent_name} session={session_id} stage={stage}"
        self.logger.info(f"{label} result=started path={path}")
        try:
            before = await asyncio.to_thread(self._read_optional, path)

            async def observe(chunk):
                publish(chunk)

            with self.agent_runtime.live_turns.observe(agent_name, session_id, query) as publish:
                reply = await collect_reply(
                    self.agent_runtime.agent_wrapper.reply_stream(
                        query,
                        resume=session_id,
                        activity_source="proactive",
                        activity_task_id=agent_name,
                        **options,
                    ),
                    on_chunk=observe,
                )
            if reply["status"] != "completed":
                raise RuntimeError(reply.get("error") or f"Proactive agent {reply['status']}")
            result = await complete_in_thread(check)
            after = await asyncio.to_thread(self._read_optional, path)
            change = "unchanged" if before == after else "created" if before is None else "updated"
            action = frontmatter.loads(after).get("action") if after is not None else None
            explanation = frontmatter.loads(after).content if action == "skip" else reply["result"]
            explanation = " ".join(explanation.split())[:500]
            self.logger.info(
                f"{label} result={change} action={action} elapsed={monotonic() - started:.2f}s "
                f"path={path} bytes={len((after or '').encode('utf-8'))} summary={explanation!r}",
            )
            return result
        except BaseException as exc:
            outcome = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
            self.logger.error(
                f"{label} result={outcome} elapsed={monotonic() - started:.2f}s path={path} error={exc}",
            )
            raise

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"agent_name"}:
            raise ValueError("proactive_agent_step requires only agent_name")
        self.require_paths(self.workspace_path)
        async with asyncio.timeout(self.timeout_seconds):
            self.context.response.answer = await self.agent_runtime.run(arguments["agent_name"], self._execute)
        return self.context.response

    async def _execute(self, agent, events):
        """Retain records and summarize only when a fresh run generates a reminder."""
        runtime = self.agent_runtime
        now = runtime.now()
        paths = await asyncio.to_thread(self._paths, agent, now)
        summary_before = await asyncio.to_thread(self._read_optional, paths["today_summary_path"])
        if not (await asyncio.to_thread(runtime.files.read, paths["task_path"])).strip():
            raise ValueError("TASK.md must not be blank")
        options = self.session_options(agent, paths, now)
        session_id = await runtime.sessions.ensure(
            agent.agent_name,
            runtime.agent_wrapper,
            options,
            now,
            new=True,
        )
        query = await asyncio.to_thread(self._record_query, agent, events)

        async def run(stage, path, check):
            prompt = (
                query
                if stage == "record"
                else self.prompt_format(
                    f"query_{stage}",
                    **{key: str(value) for key, value in paths.items()},
                )
            )
            return await self._run_stage(agent.agent_name, session_id, stage, prompt, path, check, options)

        document = await run(
            "record",
            paths["record_path"],
            lambda: self._read_document(paths["record_path"], allow_empty_skip=True),
        )
        if document["action"] == "skip":
            if await asyncio.to_thread(self._read_optional, paths["today_summary_path"]) != summary_before:
                raise ValueError("a skipped record must not change the daily summary")
            await complete_in_thread(runtime.files.remove, paths["record_path"])
            for stage in ("update", "reminder", "summary"):
                key = {
                    "update": "record_path",
                    "reminder": "reminder_path",
                    "summary": "today_summary_path",
                }[stage]
                self.logger.info(
                    f"Proactive agent={agent.agent_name} session={session_id} stage={stage} "
                    f"result=skipped reason=record_skipped path={paths[key]}",
                )
            return ProactiveRunResult(notifications=[]), session_id

        await run(
            "update",
            paths["record_path"],
            lambda: self._read_document(paths["record_path"], action="write"),
        )
        record = await asyncio.to_thread(runtime.files.read, paths["record_path"])
        result = await run("reminder", paths["reminder_path"], lambda: self._prepare_reminder(paths))
        if not result.notifications:

            def discard_draft():
                if self._read_optional(paths["today_summary_path"]) != summary_before:
                    raise ValueError("a skipped reminder must not change the daily summary")
                runtime.files.remove(paths["record_path"])
                runtime.files.remove(paths["reminder_path"])

            await complete_in_thread(discard_draft)
            self.logger.info(
                f"Proactive agent={agent.agent_name} session={session_id} stage=summary "
                f"result=skipped reason=reminder_skipped path={paths['today_summary_path']}",
            )
            return result, session_id
        reminder = await asyncio.to_thread(runtime.files.read, paths["reminder_path"])
        result = await run(
            "summary",
            paths["today_summary_path"],
            lambda: self._read_artifacts(paths, summary_before, record, reminder),
        )
        return result, session_id


@R.register("proactive_query_step")
class ProactiveQueryStep(BaseAgentStep):
    """Fork the latest queryable session and ask directly, without a mode prompt."""

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"agent_name", "query"} or not str(arguments["query"]).strip():
            raise ValueError("proactive_query_step requires agent_name and query")
        self.require_paths(self.workspace_path)
        reply = await self.query_reply(arguments["agent_name"], arguments["query"])
        if reply["status"] != "completed":
            raise RuntimeError(reply.get("error") or f"Proactive query {reply['status']}")
        self.context.response.answer = {"agent_name": arguments["agent_name"], "answer": reply["result"]}
        return self.context.response
