"""Consume one event batch in a fresh session and persist a durable completion receipt."""

import asyncio
from datetime import timedelta
import math
from time import monotonic

import frontmatter

from .base import BaseAgentStep
from ....components import R
from ....enumeration import ChunkEnum
from ....schema.proactive_agent import ProactiveRunResult
from ....schema import TokenUsage, collect_reply
from ....utils.async_utils import complete_in_thread
from ....utils.counter import global_counter_add_many


@R.register("proactive_agent_step")
class ProactiveAgentStep(BaseAgentStep):
    """Restore context, process inputs and publish validated reminder documents."""

    def __init__(self, timeout_seconds=1200, **kwargs):
        super().__init__(**kwargs)
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = timeout_seconds
        self._round_files = {}

    def _paths(self, agent, now):
        runtime = self.agent_runtime
        daily = runtime.files.path(self.config_value("daily_dir"))
        reminder_dir = runtime.agents.directory_for(agent) / "reminder"
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

    def _query(self, agent, events):
        if agent.agent_type == "cron":
            return self.prompt_format("cron_query")
        files = self.agent_runtime.files
        inputs = []
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
            "proactive_query",
            input_content="\n\n".join(inputs),
        )

    def _read_optional(self, path):
        """Only a missing optional file is treated as absent."""
        try:
            return self.agent_runtime.files.read(path)
        except FileNotFoundError:
            return None

    def _read_document(self, path, *, record=False):
        """Require an explicit decision and validate the body."""
        document = frontmatter.loads(self.agent_runtime.files.read(path))
        if document.get("action") not in ("write", "skip"):
            raise ValueError(f"{path}: action must be write or skip")
        if record and document["action"] == "skip":
            if document.content.strip():
                raise ValueError(f"{path}: a skipped record body must be blank")
        elif not document.content.strip():
            raise ValueError(f"{path}: document body must not be blank")
        return document

    def _read_artifacts(self, agent, paths, summary_before):
        """Validate the final branch and derive notifications only from saved documents."""
        files = self.agent_runtime.files
        record = self._read_document(paths["record_path"], record=True)
        if agent.agent_type == "cron" and record["action"] != "write":
            raise ValueError("a cron task must write its execution record")
        notifications = []
        if record["action"] == "skip":
            if self._read_optional(paths["reminder_path"]) is not None:
                raise ValueError("a skipped record must not create a reminder")
        else:
            reminder = self._read_document(paths["reminder_path"])
            if agent.agent_type == "cron" and (reminder["action"] != "write" or reminder.get("priority") != "P0"):
                raise ValueError("a cron task must write a P0 report")
            if reminder["action"] == "write":
                notifications.append(
                    {
                        "priority": reminder.get("priority"),
                        "message": reminder.content,
                        "paths": [
                            paths[key].relative_to(files.workspace).as_posix()
                            for key in ("reminder_path", "record_path")
                        ],
                    },
                )
        result = ProactiveRunResult(notifications=notifications)
        summary = self._read_optional(paths["today_summary_path"])
        if record["action"] == "skip":
            if summary != summary_before:
                raise ValueError("a skipped record must not change the daily summary")
        else:
            if not summary or not summary.strip():
                raise ValueError("a new record requires a nonempty daily summary")
            if summary == summary_before:
                raise ValueError("a new record must update its daily summary")
        return result

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"agent_name"}:
            raise ValueError("proactive_agent_step requires only agent_name")
        self.require_paths(self.workspace_path)
        self._round_files = {}
        self.context.response.metadata.pop("reme.agent_run", None)
        started, outcome = monotonic(), "failed"
        try:
            self.context.response.answer = await self.agent_runtime.run(
                arguments["agent_name"],
                self._execute,
                on_failure=self._restore_files,
            )
            outcome = "completed"
        except asyncio.CancelledError:
            outcome = "interrupted"
            raise
        finally:
            if stats := self.context.response.metadata.get("reme.agent_run"):
                stats.update(result=outcome, elapsed_seconds=monotonic() - started)
                prefix = ("agent_runs", stats["agent_type"])
                counts = {
                    "runs": 1,
                    outcome: 1,
                    "tool_calls": stats["tool_calls"],
                    "lookup_calls": stats["lookup_calls"],
                }
                counts["elapsed_ms"] = round(stats["elapsed_seconds"] * 1000)
                if stats["usage"] is not None:
                    counts.update(stats["usage"], usage_runs=1)
                global_counter_add_many(
                    self.app_context.metadata,
                    {(*prefix, key): value for key, value in counts.items()},
                )
        return self.context.response

    async def _restore_files(self):
        """Restore only this round's files when its input receipt was not committed."""

        def restore():
            files = self.agent_runtime.files
            for path, before in self._round_files.items():
                if self._read_optional(path) == before:
                    continue
                if before is None:
                    files.remove(path)
                else:
                    files.write(path, before)

        await complete_in_thread(restore)

    async def _execute(self, agent, events):
        """Run the complete workflow in one turn, then validate its saved decisions."""
        stats = {
            "agent_type": agent.agent_type,
            "session_id": None,
            "model_status": "not_started",
            "usage": None,
            "tool_calls": 0,
            "lookup_calls": 0,
        }
        self.context.response.metadata["reme.agent_run"] = stats
        runtime = self.agent_runtime
        now = runtime.now()
        paths = await asyncio.to_thread(self._paths, agent, now)
        summary_before = await asyncio.to_thread(self._read_optional, paths["today_summary_path"])
        self._round_files = {
            paths["record_path"]: None,
            paths["reminder_path"]: None,
            paths["today_summary_path"]: summary_before,
        }
        if not (await asyncio.to_thread(runtime.files.read, paths["task_path"])).strip():
            raise ValueError("TASK.md must not be blank")
        options = self.session_options(agent, paths, now)
        query = await asyncio.to_thread(self._query, agent, events)
        started = monotonic()
        label = f"Proactive agent={agent.agent_name} agent_type={agent.agent_type}"
        session_id = None
        calls = {}
        try:

            async def observe(chunk):
                if chunk.chunk_type in {ChunkEnum.TOOL_CALL, ChunkEnum.TOOL_RESULT}:
                    if identifier := chunk.tool_call_id or chunk.block_id or chunk.message_id:
                        key = (chunk.turn_id, identifier)
                        if chunk.tool_call_name or key in calls:
                            calls[key] = chunk.tool_call_name or calls.get(key)
                elif chunk.chunk_type == ChunkEnum.USAGE and isinstance(chunk.chunk, dict):
                    stats["usage"] = TokenUsage.from_provider(chunk.chunk).model_dump()
                elif chunk.chunk_type == ChunkEnum.REPLY_END and isinstance(chunk.chunk, dict):
                    stats["model_status"] = chunk.chunk.get("status", "failed")
                    if chunk.chunk.get("usage") is not None:
                        stats["usage"] = TokenUsage.from_provider(chunk.chunk["usage"]).model_dump()
                publish(chunk)

            model_error = None
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    session_id = await runtime.sessions.ensure(
                        agent.agent_name,
                        runtime.agent_wrapper,
                        options,
                        now,
                        new=True,
                    )
                    label += f" session={session_id}"
                    self.logger.info(f"{label} result=started path={paths['record_path']}")
                    with runtime.live_turns.observe(agent.agent_name, session_id, query) as publish:
                        reply = await collect_reply(
                            runtime.agent_wrapper.reply_stream(
                                query,
                                resume=session_id,
                                activity_source="proactive",
                                activity_task_id=agent.agent_name,
                                **options,
                            ),
                            on_chunk=observe,
                        )
                    if reply["status"] != "completed":
                        raise RuntimeError(reply.get("error") or f"Proactive agent {reply['status']}")
            except Exception as exc:
                model_error = exc
                stats["model_status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
            try:
                result = await asyncio.to_thread(self._read_artifacts, agent, paths, summary_before)
            except Exception as validation_error:
                if model_error is not None:
                    raise model_error from validation_error
                raise
            if model_error is not None:
                self.logger.warning(f"{label} result=accepted_artifacts model_error={type(model_error).__name__}")
            self.logger.info(
                f"{label} result=validated notifications={len(result.notifications)} "
                f"elapsed={monotonic() - started:.2f}s path={paths['record_path']}",
            )
        except BaseException as exc:
            outcome = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
            if outcome == "interrupted":
                stats["model_status"] = outcome
            self.logger.error(f"{label} result={outcome} elapsed={monotonic() - started:.2f}s error={exc}")
            raise
        finally:
            stats.update(
                session_id=session_id,
                tool_calls=len(calls),
                lookup_calls=sum(name in {"webSearch", "web_search", "web_extract"} for name in calls.values()),
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
        self.context.response.answer = {
            "agent_name": arguments["agent_name"],
            "answer": reply["result"],
        }
        return self.context.response
