"""Consume one event batch in a fresh session and persist a durable completion receipt."""

import asyncio
from datetime import timedelta
import json
import math
from posixpath import join, normpath
import shlex
from time import monotonic
from typing import get_args
from urllib.parse import unquote, urlsplit

import frontmatter

from .base import BaseAgentStep
from ._artifacts import RoundArtifacts
from ....core.registry import R
from ....enumeration import ChunkEnum
from ....schema.agent_runtime import InboxPriority
from ....schema.proactive_agent import CRON_EVENT_SOURCE, ProactiveRunResult
from ....schema import TokenUsage, collect_reply
from ....utils.async_utils import complete_in_thread
from ....utils.counter import global_counter_add_many
from ....utils.documents.markdown_parser import markdown_link_targets


@R.register("proactive_agent_step")
class ProactiveAgentStep(BaseAgentStep):
    """Restore context, process inputs and publish validated reminder documents."""

    def __init__(self, timeout_seconds=1200, max_local_search_calls=10, max_timeline_entries=10, **kwargs):
        super().__init__(**kwargs)
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = timeout_seconds
        for name, value in (
            ("max_local_search_calls", max_local_search_calls),
            ("max_timeline_entries", max_timeline_entries),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_local_search_calls = max_local_search_calls
        self.max_timeline_entries = max_timeline_entries
        self._round_files = None

    def _paths(self, agent, now):
        runtime = self.agent_runtime
        daily = runtime.files.path(self.config_value("daily_dir"))
        agent_dir = runtime.agents.directory_for(agent)
        summary_dir = agent_dir / "summary"
        reminder_dir = agent_dir / "reminder"
        today = now.date()
        yesterday = today - timedelta(days=1)
        day = today.isoformat()
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
            "today_summary_path": runtime.files.checked(summary_dir / f"{today}.md"),
            "yesterday_summary_path": runtime.files.checked(summary_dir / f"{yesterday}.md"),
            "record_path": detail,
            "reminder_path": reminder,
        }

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

    def _query(self, events, paths, now, task_content, *, is_cron, **prompt_values):
        """Build the task type's Markdown query with its task snapshot and round inputs."""
        files = self.agent_runtime.files
        inputs = []
        for number, event in enumerate(() if is_cron else events, 1):
            attachments = event["paths"]
            inputs.append(
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
        # Cron keeps its planned time in the inbox. The prompt uses the actual run time.
        return self.prompt_format(
            "cron_query" if is_cron else "proactive_query",
            workspace_dir=self.workspace_path,
            app_timezone=self.app_context.app_config.timezone or str(now.tzinfo),
            task_content=task_content,
            input_content="\n\n".join(inputs),
            now=now.isoformat(),
            max_local_search_calls=self.max_local_search_calls,
            max_timeline_entries=self.max_timeline_entries,
            **prompt_values,
            **{
                key: value.relative_to(files.workspace).as_posix() for key, value in paths.items() if key != "task_path"
            },
        )

    def _read_optional(self, path):
        """Only a missing optional file is treated as absent."""
        try:
            return self.agent_runtime.files.read(path)
        except FileNotFoundError:
            return None

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

    def _read_document(self, path, *, record=False):
        """Require an explicit decision and validate the body."""
        document = frontmatter.loads(self.agent_runtime.files.read(path))
        action = document.get("action")
        actions = ("write", "skip") if record else (*get_args(InboxPriority), "skip")
        if action not in actions:
            raise ValueError(f"{path}: action must be one of {', '.join(actions)}")
        if not record and set(document.metadata) != {"action"}:
            raise ValueError(f"{path}: reminder frontmatter must contain only action")
        if record and action == "skip":
            if document.content.strip():
                raise ValueError(f"{path}: a skipped record body must be blank")
        elif not document.content.strip():
            raise ValueError(f"{path}: document body must not be blank")
        return document

    def _warn_local_links(self, path, content, before=""):
        """Check new Markdown links relative to their source without blocking submission."""
        files = self.agent_runtime.files
        directory = path.parent.relative_to(files.workspace).as_posix()
        try:
            targets = markdown_link_targets(content) - markdown_link_targets(before)
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

    def _read_artifacts(self, paths, summary_before, *, is_cron):
        """Validate the final branch and derive notifications only from saved documents."""
        files = self.agent_runtime.files
        record = self._read_document(paths["record_path"], record=True)
        summary = self._read_optional(paths["today_summary_path"])
        if record["action"] == "skip":
            if is_cron:
                raise ValueError("a scheduled task requires an execution record and daily summary update")
            if self._read_optional(paths["reminder_path"]) is not None:
                raise ValueError("a skipped record must not create a reminder")
            if summary != summary_before:
                raise ValueError("a skipped record must not change the daily summary")
            return ProactiveRunResult(notifications=[])

        reminder = self._read_document(paths["reminder_path"])
        if not summary or not summary.strip():
            raise ValueError("a new record requires a nonempty daily summary")
        if summary == summary_before:
            raise ValueError("a new record must update its daily summary")
        notifications = []
        if reminder["action"] != "skip":
            notifications.append(
                {
                    "priority": reminder["action"],
                    "message": reminder.content,
                    "paths": [
                        paths[key].relative_to(files.workspace).as_posix() for key in ("reminder_path", "record_path")
                    ],
                },
            )
        result = ProactiveRunResult(notifications=notifications)
        self._warn_local_links(paths["record_path"], record.content)
        self._warn_local_links(paths["reminder_path"], reminder.content)
        self._warn_local_links(paths["today_summary_path"], summary, summary_before or "")
        return result

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"agent_name"}:
            raise ValueError("proactive_agent_step requires only agent_name")
        self.require_paths(self.workspace_path)
        self._round_files = None
        self.context.response.metadata.pop("reme.agent_run", None)
        started, outcome = monotonic(), "failed"
        try:
            self.context.response.answer = await self.agent_runtime.run(
                arguments["agent_name"],
                self._execute,
                on_failure=self._restore_files,
                on_write=self._record_write,
                before_write=self._check_write,
            )
            outcome = "completed"
        except asyncio.CancelledError:
            outcome = "interrupted"
            raise
        finally:
            if self._round_files is not None:
                try:
                    await complete_in_thread(self._round_files.close)
                except OSError as exc:
                    self.logger.warning(f"Could not remove round backup: {exc}")
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

    async def _restore_files(self):
        """Restore only this round's files when its input receipt was not committed."""
        if self._round_files is not None:
            await self._round_files.rollback()

    def _record_write(self, path, version):
        """Track the outbox's own canonicalization without claiming external edits."""
        self._round_files.record_write(path, version)

    def _check_write(self, path):
        """Keep user edits authoritative until the outbox has added its metadata."""
        self._round_files.check_written(path)

    def _make_artifacts(self, paths):
        return RoundArtifacts(self.agent_runtime.files, paths)

    async def _execute(self, agent, events):
        """Run the complete workflow in one turn, then validate its saved decisions."""
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
        runtime = self.agent_runtime
        now = runtime.now()
        paths = await asyncio.to_thread(self._paths, agent, now)
        artifacts = self._round_files = self._make_artifacts(paths)
        before = await artifacts.prepare()
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
        turn_id, warned = None, False

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

        async def observe(chunk):
            nonlocal turn_id
            turn_id = chunk.turn_id or turn_id
            kind, payload = chunk.chunk_type, chunk.chunk
            if kind in {ChunkEnum.TOOL_CALL, ChunkEnum.TOOL_RESULT}:
                identifier = chunk.tool_call_id or chunk.block_id or chunk.message_id
                if identifier and (chunk.tool_call_name or identifier in calls):
                    call = calls.setdefault(identifier, {})
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
                    stats["usage"] = TokenUsage.from_provider(payload).model_dump()
            publish(chunk)

        try:
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
                        try:
                            reply = await collect_reply(
                                runtime.agent_wrapper.reply_stream(query, resume=session_id, **options),
                                on_chunk=observe,
                            )
                        finally:
                            async with artifacts.locked():
                                await complete_in_thread(artifacts.capture_outputs)
                    if reply["status"] != "completed":
                        raise RuntimeError(reply.get("error") or f"Proactive agent {reply['status']}")
            except Exception as exc:
                model_error = exc
                stats["model_status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
            try:
                async with artifacts.locked():
                    result = await asyncio.to_thread(self._read_artifacts, paths, before, is_cron=is_cron)
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
            self.logger.error(
                f"{label} result={outcome} elapsed={monotonic() - started:.2f}s error={type(exc).__name__}: {exc}",
            )
            raise
        finally:
            if session_id is not None:
                try:
                    saved = await runtime.agent_wrapper.get_tool_calls(session_id, turn_id)
                    if saved is not None:
                        for call in saved:
                            previous = calls.get(call["id"], {})
                            calls[call["id"]] = {
                                **previous,
                                **call,
                                "actions": call.get("actions") or previous.get("actions", []),
                            }
                        stats["tool_call_source"] = "native"
                except Exception as exc:
                    self.logger.warning(f"{label} could not read saved tool statistics: {exc}")
            stats["session_id"] = session_id
            update_counts()
        return result, session_id
