"""Consume one event batch in a fresh session and persist a durable completion receipt."""

import asyncio
from datetime import datetime
from functools import partial
import json
import math
from pathlib import Path
from posixpath import join, normpath
import shlex
from time import monotonic
from urllib.parse import unquote, urlsplit

import yaml
from pydantic import ValidationError

from .base import BaseAgentStep
from ....core.registry import R
from ....enumeration import ChunkEnum
from ....schema import TokenUsage, collect_reply
from ....schema.proactive_runtime import AgentReminder
from ....schema.proactive_agent import CRON_EVENT_SOURCE, ProactiveRecordDecision
from ....utils.async_utils import complete_in_thread
from ....utils.counter import global_counter_add_many
from ....utils.documents.markdown_parser import markdown_link_targets
from ....utils.files.path_lock import get_path_lock
from ....utils.proactive_record import RoundRecord, split_record

_NO_KEY_INFORMATION = "NO_KEY_INFORMATION"


@R.register("proactive_agent_step")
class ProactiveAgentStep(BaseAgentStep):
    """Process one round and record its results and reminder decision together."""

    def __init__(self, timeout_seconds=1200, max_local_search_calls=10, recent_history_hours=24, **kwargs):
        super().__init__(**kwargs)
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = timeout_seconds
        for name, value in (
            ("max_local_search_calls", max_local_search_calls),
            ("recent_history_hours", recent_history_hours),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_local_search_calls = max_local_search_calls
        self.recent_history_hours = recent_history_hours

    def _paths(self, agent, now: datetime) -> dict[str, Path]:
        runtime = self.proactive_runtime
        daily = runtime.files.path(self.config_value("daily_dir"))
        day = daily / now.date().isoformat()
        stem = f"{agent.agent_name}_{now:%H%M%S}"
        record = runtime.files.checked(day / f"{stem}.md")
        collision = 0
        while record.exists():
            collision += 1
            record = runtime.files.checked(day / f"{stem}_{collision}.md")
        return {"daily_dir": daily, "task_path": runtime.agents.task_path(agent), "record_path": record}

    def session_options(self, agent):
        """Build tool bindings and permissions for an automatic run."""
        options = {
            "config": {"skills.include_instructions": False},
            "job_tools": self.kwargs.get("job_tools") or [],
            "injected_job_kwargs": {"_agent_name": agent.agent_name},
            "approval_mode": "deny_all",
        }
        if (sandbox := self.kwargs.get("sandbox", "workspace-write")) is not None:
            options["sandbox"] = sandbox
        if (effort := self.kwargs.get("effort")) is not None:
            options["effort"] = effort
        return options

    def _query(self, events, paths, now, task_content, *, is_cron):
        """Build the round prompt from the task snapshot and claimed events."""
        files = self.proactive_runtime.files
        event_sections = []
        for number, event in enumerate(events, 1):
            if is_cron:
                event_sections.append(self.prompt_format("scheduled_event_input", event_ts=event["event_ts"]))
                continue
            attachments = event["paths"]
            event_sections.append(
                self.prompt_format(
                    "event_input",
                    number=number,
                    event_ts=event["event_ts"],
                    source=event["source"],
                    description=event["description"],
                    paths="\n".join(
                        f"- {files.path(path).relative_to(files.workspace).as_posix()}" for path in attachments
                    ),
                    has_files=bool(attachments),
                    no_files=not attachments,
                ),
            )
        return self.prompt_format(
            "task_query",
            workspace_dir=self.workspace_path,
            app_timezone=self.app_context.app_config.timezone or str(now.tzinfo),
            event_batch="\n\n".join(event_sections),
            is_cron=is_cron,
            is_event=not is_cron,
            now=now.isoformat(),
            max_local_search_calls=self.max_local_search_calls,
            task_description=task_content,
            recent_history_hours=self.recent_history_hours,
            no_key_information=_NO_KEY_INFORMATION,
            daily_dir=paths["daily_dir"].relative_to(files.workspace).as_posix(),
            record_path=paths["record_path"].relative_to(files.workspace).as_posix(),
        )

    @staticmethod
    def _lookup_type(call):
        """Count named tools, native searches and explicit rg/grep shell commands."""
        name = (call.get("name") or "").rsplit("__", 1)[-1]
        if name in {"webSearch", "web_search", "web_extract"}:
            return "web_lookup_calls"
        if name in {"rg", "grep", "search", "file_search", "search_files"} or any(
            action.get("type") == "search" for action in call.get("actions", [])
        ):
            return "local_search_calls"
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        command = (arguments.get("cmd") or arguments.get("command")) if isinstance(arguments, dict) else None
        if not isinstance(command, str):
            return None
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()\n")
        lexer.whitespace, lexer.whitespace_split = " \t\r", True
        try:
            tokens = list(lexer)
        except ValueError:
            return None
        separators = {";", "&&", "||", "|", "&", "(", ")", "\n", "then", "do"}
        for index, token in enumerate(tokens):
            tool = token.rsplit("/", 1)[-1]
            if tool not in {"rg", "grep"} or (index and tokens[index - 1] not in separators):
                continue
            end = next((i for i in range(index + 1, len(tokens)) if tokens[i] in separators), len(tokens))
            if tool == "grep" or "--files" not in tokens[index + 1 : end]:
                return "local_search_calls"
        return None

    def _warn_local_links(self, path, content):
        """Check Markdown links relative to their source without blocking submission."""
        files = self.proactive_runtime.files
        directory = path.parent.relative_to(files.workspace).as_posix()
        try:
            targets = markdown_link_targets(content)
        except (ValueError, RuntimeError) as exc:
            self.logger.warning(f"Could not parse Markdown links source={path}: {exc}")
            return
        for target in sorted(targets):
            try:
                url = urlsplit(target)
                if url.scheme or url.netloc or not url.path:
                    continue
                relative_path = normpath(join(directory, unquote(url.path)))
                if files.path(relative_path).exists():
                    continue
                reason = "target does not exist"
            except (OSError, ValueError, RuntimeError) as exc:
                reason = str(exc)
            self.logger.warning(f"Proactive local link source={path} target={target!r}: {reason}")

    def _read_record(self, path: Path, reply_text: str | None, *, is_cron: bool) -> AgentReminder | None:
        """Validate the round record or an explicit event skip."""
        files = self.proactive_runtime.files
        try:
            text = files.read(path)
        except FileNotFoundError:
            if not is_cron and reply_text is not None and reply_text.strip() == _NO_KEY_INFORMATION:
                return None
            raise
        metadata, body = split_record(text)
        decision = ProactiveRecordDecision.model_validate(metadata)
        self._warn_local_links(path, body)
        if decision.reminder_priority == "skip":
            return None
        self._warn_local_links(path, decision.reminder_message)
        return AgentReminder(
            priority=decision.reminder_priority,
            message=decision.reminder_message,
            id=path.relative_to(files.workspace).as_posix(),
        )

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"agent_name"}:
            raise ValueError("proactive_agent_step requires only agent_name")
        self.require_paths(self.workspace_path)
        record = RoundRecord(self.proactive_runtime.files)
        self.context.response.metadata.pop("reme.agent_run", None)
        started, outcome = monotonic(), "failed"
        try:
            self.context.response.answer = await self.proactive_runtime.run(
                arguments["agent_name"],
                partial(self._execute, record=record),
                record=record,
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
                    "web_lookup_calls": stats["web_lookup_calls"],
                    "local_search_calls": stats["local_search_calls"],
                    "elapsed_ms": round(stats["elapsed_seconds"] * 1000),
                }
                if stats["usage"] is not None:
                    counts.update(stats["usage"], usage_runs=1)
                global_counter_add_many(
                    self.app_context.metadata,
                    {(*prefix, key): value for key, value in counts.items()},
                )
        return self.context.response

    async def _execute(self, agent, events, *, record):
        """Run the workflow and allow one follow-up to repair the saved record."""
        is_cron = events[0]["source"] == CRON_EVENT_SOURCE
        agent_type = "cron" if is_cron else "proactive"
        stats = {
            "agent_type": agent_type,
            "session_id": None,
            "model_status": "not_started",
            "usage": None,
            "tool_calls": 0,
            "web_lookup_calls": 0,
            "local_search_calls": 0,
            "max_local_search_calls": self.max_local_search_calls,
        }
        self.context.response.metadata["reme.agent_run"] = stats
        runtime = self.proactive_runtime
        now = runtime.now()
        paths = await asyncio.to_thread(self._paths, agent, now)
        await record.prepare(paths["record_path"])
        task_content = await asyncio.to_thread(runtime.files.read, paths["task_path"])
        if not task_content.strip():
            raise ValueError("TASK.md must not be blank")
        options = self.session_options(agent)
        query = await asyncio.to_thread(
            self._query,
            events,
            paths,
            now,
            task_content,
            is_cron=is_cron,
        )
        started = monotonic()
        label = f"Proactive agent={agent.agent_name} agent_type={agent_type}"
        session_id = None
        calls = {}
        turn_ids, turn_usage = {}, {}
        warned = False

        def update_counts():
            nonlocal warned
            stats["tool_calls"] = len(calls)
            kinds = [self._lookup_type(call) for call in calls.values()]
            for key in ("web_lookup_calls", "local_search_calls"):
                stats[key] = kinds.count(key)
            if not warned and stats["local_search_calls"] > self.max_local_search_calls:
                self.logger.warning(
                    f"{label} local_search_calls={stats['local_search_calls']} "
                    f"exceeds limit={self.max_local_search_calls}; continuing",
                )
                warned = True

        async def observe(chunk, *, attempt, publish):
            if chunk.turn_id:
                turn_ids[attempt] = chunk.turn_id
            kind, payload = chunk.chunk_type, chunk.chunk
            if kind in {ChunkEnum.TOOL_CALL, ChunkEnum.TOOL_RESULT}:
                identifier = chunk.tool_call_id or chunk.block_id or chunk.message_id
                key = (attempt, identifier)
                if identifier and (chunk.tool_call_name or key in calls):
                    call = calls.setdefault(key, {})
                    call["name"] = chunk.tool_call_name or call.get("name")
                    if kind == ChunkEnum.TOOL_CALL or (
                        call["name"] == "commandExecution"
                        and isinstance(payload, dict)
                        and ("command" in payload or "commandActions" in payload)
                    ):
                        if isinstance(payload, dict):
                            call["arguments"] = payload
                            call["actions"] = payload.get("commandActions") or call.get("actions", [])
                        elif isinstance(payload, str):
                            previous = call.get("arguments")
                            call["arguments"] = (
                                previous + payload if chunk.event == "delta" and isinstance(previous, str) else payload
                            )
                    if chunk.event != "delta":
                        update_counts()
            elif kind in {ChunkEnum.USAGE, ChunkEnum.REPLY_END} and isinstance(payload, dict):
                if kind == ChunkEnum.REPLY_END:
                    stats["model_status"] = payload.get("status", "failed")
                    payload = payload.get("usage")
                if payload is not None:
                    turn_usage[attempt] = TokenUsage.from_provider(payload)
            publish(chunk)

        deadline = asyncio.get_running_loop().time() + self.timeout_seconds
        try:
            for attempt in range(2):
                turn_ids[attempt] = None
                reply = model_error = None
                try:
                    async with asyncio.timeout_at(deadline):
                        if attempt == 0:
                            session_id = await runtime.sessions.ensure(
                                agent.agent_name,
                                runtime.agent_wrapper,
                                options,
                                now,
                                new=True,
                            )
                            label += f" session={session_id}"
                            self.logger.info(f"{label} result=started path={record.path}")
                        with runtime.live_turns.observe(agent.agent_name, session_id, query) as publish:
                            try:
                                reply = await collect_reply(
                                    runtime.agent_wrapper.reply_stream(query, resume=session_id, **options),
                                    on_chunk=partial(observe, attempt=attempt, publish=publish),
                                )
                            finally:
                                if attempt == 0:
                                    async with await get_path_lock(record.path):
                                        await complete_in_thread(record.capture)
                        if reply["status"] != "completed":
                            raise RuntimeError(reply.get("error") or f"Proactive agent {reply['status']}")
                except Exception as exc:
                    model_error = exc
                    stats["model_status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
                if attempt and model_error is not None:
                    raise model_error
                try:
                    async with await get_path_lock(record.path):
                        if attempt:
                            decision = ProactiveRecordDecision.model_validate(reply["structured_output"])
                            await complete_in_thread(record.replace_frontmatter, decision.model_dump())
                        else:
                            await asyncio.to_thread(record.check_version, record.path)
                        reminder = await asyncio.to_thread(
                            self._read_record,
                            record.path,
                            reply["result"] if attempt == 0 and model_error is None else None,
                            is_cron=is_cron,
                        )
                except Exception as validation_error:
                    if model_error is not None:
                        raise model_error from validation_error
                    if attempt or not isinstance(validation_error, (yaml.YAMLError, ValidationError)):
                        raise
                    record_path = record.path.relative_to(runtime.workspace_path).as_posix()
                    query = self.prompt_format(
                        "repair_record_query",
                        record_path=record_path,
                        error=str(validation_error)[:2000],
                    )
                    options = {
                        **options,
                        "sandbox": "read-only",
                        "job_tools": [],
                        "output_schema": ProactiveRecordDecision.model_json_schema(),
                    }
                    self.logger.info(f"{label} result=repairing_record path={record.path}")
                else:
                    break
            if model_error is not None:
                self.logger.warning(f"{label} result=accepted_record model_error={type(model_error).__name__}")
            self.logger.info(
                f"{label} result=validated notifications={int(reminder is not None)} "
                f"elapsed={monotonic() - started:.2f}s path={paths['record_path']}",
            )
        except BaseException as exc:
            outcome = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
            if outcome == "interrupted":
                stats["model_status"] = outcome
            self.logger.error(
                f"{label} result={outcome} elapsed={monotonic() - started:.2f}s error={type(exc).__name__}: {exc}",
            )
            raise
        finally:
            if session_id is not None:
                for attempt, turn_id in turn_ids.items():
                    if turn_id is None and len(turn_ids) > 1:
                        # An unfiltered native read would include both turns and count calls twice.
                        continue
                    try:
                        saved = await runtime.agent_wrapper.get_tool_calls(session_id, turn_id)
                        if saved is not None:
                            for call in saved:
                                key = (attempt, call["id"])
                                previous = calls.get(key, {})
                                calls[key] = {
                                    **previous,
                                    **call,
                                    "actions": call.get("actions") or previous.get("actions", []),
                                }
                            stats["tool_call_source"] = "native"
                    except Exception as exc:
                        self.logger.warning(f"{label} could not read saved tool statistics: {exc}")
            stats["session_id"] = session_id
            if turn_usage:
                stats["usage"] = TokenUsage.combine(list(turn_usage.values())).model_dump()
            update_counts()
        return reminder, session_id
