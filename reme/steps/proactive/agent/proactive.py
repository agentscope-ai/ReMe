"""Consume one event batch through a resumed agent and durable completion receipt."""

import asyncio
from datetime import date
import math

import frontmatter

from .base import BaseAgentStep
from ....components import R
from ....schema.proactive_agent import ProactiveRunResult
from ....schema import collect_reply
from ....utils.async_utils import finish_cleanup


@R.register("proactive_agent_step")
class ProactiveAgentStep(BaseAgentStep):
    """Restore context, process inputs and publish validated reminder documents."""

    def __init__(self, reminder_history_limit: int = 10, timeout_seconds=1200, **kwargs):
        super().__init__(**kwargs)
        if type(reminder_history_limit) is not int or reminder_history_limit < 1:
            raise ValueError("reminder_history_limit must be a positive integer")
        self.reminder_history_limit = reminder_history_limit
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = timeout_seconds

    def _paths(self, agent, now):
        runtime = self.agent_runtime
        daily = runtime.files.path(self.config_value("daily_dir"))
        agent_dir = runtime.agents.directory_for(agent)
        reminder_dir = agent_dir / "reminder"
        day = now.strftime("%Y-%m-%d")
        summary = runtime.files.checked(daily / day / f"{agent.agent_id}.md")
        history = summary if summary.exists() else None
        if history is None:
            for candidate in sorted(daily.glob(f"*/{agent.agent_id}.md"), reverse=True):
                name = candidate.parent.name
                try:
                    valid_date = date.fromisoformat(name).isoformat() == name
                except ValueError:
                    continue
                if valid_date and name < day:
                    history = runtime.files.checked(candidate)
                    break
        stem = f"{agent.agent_id}_{now:%H%M%S}"
        collision = 0
        while True:
            filename = f"{stem}{'_' + str(collision) if collision else ''}.md"
            detail = runtime.files.checked(daily / day / filename)
            reminder = runtime.files.checked(reminder_dir / day / filename)
            if not detail.exists() and not reminder.exists():
                break
            collision += 1
        return {
            "daily_dir": daily,
            "reminder_dir": reminder_dir,
            "memory_path": agent_dir / "MEMORY.md",
            "summary_path": summary,
            "history_summary_path": history,
            "detail_path": detail,
            "reminder_path": reminder,
            "reminder_pattern": reminder_dir / "*" / "*.md",
        }

    def _options(self, agent):
        """Scheduled tasks have web access; event monitoring stays offline."""
        return {
            **self.web_options(agent.trigger_type == "cron"),
            "job_tools": [
                "agent_memory",
                *(["web_extract"] if agent.trigger_type == "cron" else []),
            ],
            "injected_job_kwargs": {"_agent_id": agent.agent_id},
            "sandbox": "workspace-write",
            "approval_mode": "deny_all",
        }

    def session_options(self, agent):
        """Use identical session identity and permissions for scheduled and user turns."""
        now = self.agent_runtime.now()
        system = self.prompt_format(
            "system_prompt",
            workspace_dir=self.workspace_path,
            agent_id=agent.agent_id,
            memory_path=self.agent_runtime.agents.directory_for(agent) / "MEMORY.md",
            app_timezone=self.app_context.app_config.timezone or str(now.tzinfo),
            reminder_policy=self.get_prompt("reminder_policy"),
            event=agent.trigger_type == "event",
            cron=agent.trigger_type == "cron",
        )
        return {"system_prompt": system, **self.memory_options(agent.agent_id), **self._options(agent)}

    def _query(self, agent, events, paths, now):
        cron = agent.trigger_type == "cron"
        files = self.agent_runtime.files
        inputs = []
        for number, event in enumerate(events, 1):
            if cron:
                content = self.prompt_format("cron_input", event_ts=event["event_ts"])
            else:
                content = self.prompt_format(
                    "event_input",
                    number=number,
                    event_ts=event["event_ts"],
                    source=event["source"],
                    description=event["description"],
                    paths="\n".join(f"  - `{files.path(path)}`" for path in event["paths"]),
                )
            inputs.append(content)
        return self.prompt_format(
            "query",
            **{key: str(value) for key, value in paths.items()},
            now=now.isoformat(),
            input_content="\n\n".join(inputs),
            reminder_history_limit=self.reminder_history_limit,
            first_run=paths["history_summary_path"] is None,
            has_history=paths["history_summary_path"] is not None,
            event=not cron,
            cron=cron,
        )

    def _read_artifacts(self, reply, paths, summary_mtime):
        """Build runtime notifications from saved documents, never from model response JSON."""
        if reply["status"] != "completed":
            raise RuntimeError(reply.get("error") or f"Proactive agent {reply['status']}")
        files = self.agent_runtime.files
        if not files.read(paths["summary_path"]).strip() or paths["summary_path"].stat().st_mtime_ns == summary_mtime:
            raise ValueError("proactive round must update its daily summary")
        if not paths["reminder_path"].exists():
            return ProactiveRunResult(notifications=[])
        document = frontmatter.loads(files.read(paths["reminder_path"]))
        if not files.read(paths["detail_path"]).strip():
            raise ValueError("reminder must have a nonempty detail file")
        return ProactiveRunResult(
            notifications=[
                {
                    "priority": document.get("priority"),
                    "message": document.content,
                    "paths": [
                        path.relative_to(files.workspace).as_posix()
                        for path in (paths["reminder_path"], paths["detail_path"])
                    ],
                },
            ],
        )

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"agent_id"}:
            raise ValueError("proactive_agent_step requires only agent_id")
        self.require_paths(self.workspace_path)
        self.context.response.answer = await self.agent_runtime.run(arguments["agent_id"], self._execute)
        return self.context.response

    async def _execute(self, agent, events):
        """Render a round, invoke the Harness, and validate the saved model artifacts."""
        runtime = self.agent_runtime
        now = runtime.now()
        paths = await asyncio.to_thread(self._paths, agent, now)
        summary_mtime = paths["summary_path"].stat().st_mtime_ns if paths["summary_path"].exists() else None
        if not (await asyncio.to_thread(runtime.files.read, paths["memory_path"])).strip():
            raise ValueError("proactive Agent MEMORY.md must not be blank")
        options = self.session_options(agent)
        session_id = await runtime.sessions.ensure(
            agent.agent_id,
            runtime.agent_wrapper,
            options,
            now,
        )
        query = self._query(agent, events, paths, now)

        async def observe(chunk):
            publish(chunk)

        with runtime.live_turns.observe(agent.agent_id, session_id, query) as publish:
            async with asyncio.timeout(self.timeout_seconds):
                reply = await collect_reply(
                    runtime.agent_wrapper.reply_stream(
                        query,
                        resume=session_id,
                        activity_source="proactive",
                        activity_task_id=agent.agent_id,
                        **options,
                    ),
                    on_chunk=observe,
                )
        result = await asyncio.to_thread(self._read_artifacts, reply, paths, summary_mtime)
        return result, reply["session_id"]


@R.register("proactive_query_step")
class ProactiveQueryStep(BaseAgentStep):
    """Fork the newest completed execution and ask directly, without a mode prompt."""

    async def execute(self):
        runtime = self.agent_runtime
        arguments = self.parameters()
        if set(arguments) != {"agent_id", "query"} or not str(arguments["query"]).strip():
            raise ValueError("proactive_query_step requires agent_id and query")
        self.require_paths(self.workspace_path)
        options = {
            **self.web_options(False),
            "sandbox": "read-only",
            "approval_mode": "deny_all",
        }
        async with runtime.query_turn(arguments["agent_id"]) as source_session:
            session_id = None
            try:

                async def fork():
                    nonlocal session_id
                    session_id = await runtime.agent_wrapper.create_session(
                        resume=source_session,
                        fork_session=True,
                        **options,
                    )
                    await runtime.sessions.record(arguments["agent_id"], session_id, runtime.now(), temporary=True)

                await finish_cleanup(fork())
                reply = await runtime.agent_wrapper.reply(
                    arguments["query"],
                    resume=session_id,
                    activity_source="proactive",
                    activity_task_id=arguments["agent_id"],
                    **options,
                )
                if reply["status"] != "completed":
                    raise RuntimeError(reply.get("error") or f"Proactive query {reply['status']}")
                self.context.response.answer = {
                    "agent_id": arguments["agent_id"],
                    "answer": reply["result"],
                }
                return self.context.response
            finally:
                if session_id is not None:
                    await finish_cleanup(runtime.agent_wrapper.release_session(session_id))
