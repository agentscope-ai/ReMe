"""Consume one event batch in a fresh session and persist a durable completion receipt."""

import asyncio
from datetime import datetime
import json
import math
from pathlib import Path
from posixpath import join, normpath
import shlex
from time import monotonic
from urllib.parse import unquote, urlsplit

from .base import BaseAgentStep, ActivityTracker
from ...core.registry import R
from ...enumeration import ChunkEnum
from ...schema import AgentActivity, TokenUsage, collect_reply
from ...schema.proactive_runtime import AgentReminder
from ...schema.proactive_agent import ProactiveRecordDecision
from ...utils.async_utils import complete_in_thread, finish_cleanup
from ...utils.counter import global_counter_add_many
from ...utils.documents.markdown_parser import markdown_link_targets
from ...utils.files.path_lock import get_path_lock
from ...utils.proactive_record import RoundRecord, split_record

_NO_KEY_INFORMATION = "NO_KEY_INFORMATION"


@R.register("proactive_agent_step")
class ProactiveAgentStep(BaseAgentStep):
    """Process one round and validate any enabled record and reminder output."""

    def __init__(self, timeout_seconds=1200, **kwargs):
        super().__init__(**kwargs)
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")
        self.timeout_seconds = timeout_seconds
        self._timeout = None
        self.runtime = None

    def round_options(self, paths, *, now):
        """Render invocation-owned skill content from the Runtime's stable binding."""
        runtime = self.runtime
        options = runtime.session_options(now=now)
        timezone_name = self.app_context.app_config.timezone if self.app_context else None
        for name, skill in zip(runtime.binding.skills, options["skills"], strict=True):
            if name == "proactive-record":
                skill["content"] = skill["content"].format(
                    now=now.isoformat(),
                    app_timezone=timezone_name or str(now.tzinfo),
                    daily_dir=self.to_workspace_relative(paths["daily_dir"]),
                    record_path=self.to_workspace_relative(paths["record_path"]),
                    recent_history_hours=runtime.initial_config.recent_history_hours,
                    max_local_search_calls=runtime.initial_config.max_local_search_calls,
                )
        return options

    def build_query(self, events, paths, task_content, *, is_time):
        """Render the task requirements and claimed inputs for this invocation."""
        sections = []
        for number, event in enumerate(events, 1):
            if is_time:
                sections.append(self.prompt_format("time_input", event_ts=event["event_ts"]))
                continue
            attachments = event["paths"]
            sections.append(
                self.prompt_format(
                    "event_input",
                    number=number,
                    event_id=event["id"],
                    event_ts=event["event_ts"],
                    source=event["source"],
                    description=event["description"],
                    paths="\n".join(
                        f"- {self.to_workspace_relative(self.runtime.files.path(path))}" for path in attachments
                    ),
                    has_files=bool(attachments),
                    no_files=not attachments,
                ),
            )
        record_context = {
            "has_record": "record_path" in paths,
            "daily_dir": self.to_workspace_relative(paths["daily_dir"]) if "daily_dir" in paths else "",
            "record_path": self.to_workspace_relative(paths["record_path"]) if "record_path" in paths else "",
        }
        record_instruction = (
            self.prompt_format("record_instruction", **record_context) if record_context["has_record"] else ""
        )
        return self.prompt_format(
            "query",
            event_batch="\n\n".join(sections),
            is_time=is_time,
            is_event=not is_time,
            task_description=task_content,
            record_instruction=record_instruction,
            **record_context,
        )

    def _paths(self, agent, now: datetime, receipt_paths: frozenset[str] = frozenset()) -> dict[str, Path]:
        runtime = self.runtime
        daily = runtime.files.path(self.config_value("daily_dir"))
        day = daily / now.date().isoformat()
        stem = f"{agent.agent_name}_{now:%H%M%S}"
        record = runtime.files.checked(day / f"{stem}.md")
        collision = 0
        while (
            record.relative_to(runtime.files.workspace).as_posix() in receipt_paths
            or record.exists()
            or runtime.has_record_registration(record)
        ):
            collision += 1
            record = runtime.files.checked(day / f"{stem}_{collision}.md")
        return {"daily_dir": daily, "task_path": runtime.task_path, "record_path": record}

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
        files = self.runtime.files
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

    def _read_record(
        self,
        record: RoundRecord,
        reply_text: str | None,
        *,
        is_time: bool,
    ) -> AgentReminder | None:
        """Validate the round record or an explicit event skip."""
        path = record.path
        record.check_version(path)
        files = self.runtime.files
        try:
            text = files.read(path)
        except FileNotFoundError:
            if not is_time and reply_text is not None and reply_text.strip() == _NO_KEY_INFORMATION:
                return None
            raise
        metadata, body = split_record(text)
        decision = ProactiveRecordDecision.parse(metadata)
        self._warn_local_links(path, body)
        reminder = decision.reminder(path.relative_to(files.workspace).as_posix())
        if reminder is not None:
            self._warn_local_links(path, reminder.message)
        return reminder

    async def execute(self):
        self._timeout = asyncio.timeout(self.timeout_seconds)
        async with self._timeout:
            return await self._execute_round()

    async def _discard_uncommitted_record(self, record, batch):
        # Accepted low-priority edge case: a hard exit after writing the record but before receipt commit
        # bypasses cleanup. The orphan record may suppress a reminder during history deduplication after restart.
        if record is None or record.path is None or record.written is None:
            return
        try:
            if not await self.runtime.inbox.has_receipt(item["id"] for item in batch["items"]):
                async with await get_path_lock(record.path):
                    await complete_in_thread(record.discard)
        except (OSError, ValueError, RuntimeError) as exc:
            self.logger.warning(f"Could not clean up failed round record {record.path}: {exc}")

    async def _execute_round(self):
        arguments = self.parameters()
        if set(arguments) != {"agent_name"}:
            raise ValueError("Task execution requires only agent_name")
        self.runtime = await self.agent_manager.resolve(arguments["agent_name"])
        self.require_paths(self.workspace_path)
        record = None
        self.context.response.metadata.pop("reme.agent_run", None)
        activity = None
        started, outcome = monotonic(), "failed"
        try:
            async with self.runtime.execution():
                agent = await self.runtime.definition()
                if "proactive-record" in agent.skills:
                    record = RoundRecord(self.runtime.files)
                if not agent.enabled:
                    self.context.response.answer = {"agent_name": agent.agent_name, "skipped": "disabled"}
                    return self.context.response
                batch = await self.runtime.inbox.claim(
                    limit=1 if agent.agent_type == "time" else agent.max_batch_events,
                    item_ids=self.runtime.retry_items,
                )
                if batch is None:
                    self.context.response.answer = {"agent_name": agent.agent_name, "processed": 0}
                    return self.context.response
                self.runtime.retry_items = [item["id"] for item in batch["items"]]
                self.runtime.last_started_at = self.runtime.now()
                self.runtime.activity = activity = AgentActivity(
                    session_id="",
                    turn_id=batch["token"],
                    source="task",
                    task_id=agent.agent_name,
                    phase="working",
                    started_at=self.runtime.last_started_at,
                )
                try:
                    if any((item["source"] == "time") != (agent.agent_type == "time") for item in batch["items"]):
                        raise ValueError("Inputs do not match the task trigger type")
                    reminder, session_id = await self._execute(agent, batch["items"], record=record)
                    self.context.response.answer = await finish_cleanup(
                        self.runtime.commit_round(batch, reminder, session_id, record),
                    )
                    self.runtime.activity = self.runtime.activity.model_copy(
                        update={"session_id": session_id, "phase": "completed"},
                    )
                except BaseException:
                    await finish_cleanup(self._discard_uncommitted_record(record, batch))
                    raise
                finally:
                    await finish_cleanup(self.runtime.inbox.nack(batch["token"]))
                outcome = "completed"
        except asyncio.CancelledError:
            outcome = "failed" if self._timeout.expired() else "interrupted"
            raise
        finally:
            if activity is not None and outcome != "completed":
                self.runtime.activity = self.runtime.activity.model_copy(
                    update={"phase": "failed" if outcome == "failed" else "interrupted"},
                )
            if stats := self.context.response.metadata.get("reme.agent_run"):
                if self._timeout.expired():
                    stats["model_status"] = "timeout"
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
        """Run one query with its skills and validate a record only when enabled."""
        runtime = self.runtime
        agent_type = agent.agent_type
        is_time = agent_type == "time"
        stats = {
            "agent_type": agent_type,
            "session_id": None,
            "model_status": "not_started",
            "usage": None,
            "tool_calls": 0,
            "web_lookup_calls": 0,
            "local_search_calls": 0,
            "max_local_search_calls": runtime.initial_config.max_local_search_calls,
        }
        self.context.response.metadata["reme.agent_run"] = stats
        now = runtime.now()
        paths = {"task_path": runtime.task_path}
        if record is not None:
            paths = await asyncio.to_thread(
                self._paths,
                agent,
                now,
                receipt_paths=await runtime.receipt_record_paths(),
            )
            await record.prepare(paths["record_path"])
        task_content = await asyncio.to_thread(runtime.files.read, paths["task_path"])
        if not task_content.strip():
            raise ValueError("TASK.md must not be blank")
        options = self.round_options(paths, now=now)
        query = self.build_query(events, paths, task_content, is_time=is_time)
        started = monotonic()
        label = f"Proactive agent={agent.agent_name} agent_type={agent_type}"
        session_id = turn_id = None
        usage = stream_error = None
        calls = {}
        warned = False
        tracker = ActivityTracker(self.runtime.activity)

        def update_counts():
            nonlocal warned
            stats["tool_calls"] = len(calls)
            kinds = [self._lookup_type(call) for call in calls.values()]
            for key in ("web_lookup_calls", "local_search_calls"):
                stats[key] = kinds.count(key)
            if not warned and stats["local_search_calls"] > runtime.initial_config.max_local_search_calls:
                self.logger.warning(
                    f"{label} local_search_calls={stats['local_search_calls']} "
                    f"exceeds limit={runtime.initial_config.max_local_search_calls}; continuing",
                )
                warned = True

        async def observe(chunk):
            nonlocal turn_id, usage, stream_error
            if chunk.turn_id:
                turn_id = chunk.turn_id
            kind, payload = chunk.chunk_type, chunk.chunk
            if kind == ChunkEnum.ERROR and chunk.metadata.get("will_retry") is not True:
                stream_error = str(payload)
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
                    usage = TokenUsage.from_provider(payload).model_dump()
            if self.runtime.activity is not None:
                self.runtime.activity = tracker.update(chunk)
                self.runtime.activity = self.runtime.activity.model_copy(update={"session_id": session_id or ""})
            publish(chunk)

        deadline = self._timeout.when()
        async with runtime.agent_wrapper.skill_scope(**options) as options:
            try:
                try:
                    async with asyncio.timeout_at(deadline):
                        session_id = await runtime.ensure_session(options, new=True)
                        label += f" session={session_id}"
                        self.logger.info(f"{label} result=started path={paths.get('record_path')}")
                        with runtime.live_turns.observe(agent.agent_name, session_id, query) as publish:
                            try:
                                reply = await collect_reply(
                                    runtime.agent_wrapper.reply_stream(query, resume=session_id, **options),
                                    on_chunk=observe,
                                )
                            finally:
                                if record is not None:
                                    async with await get_path_lock(record.path):
                                        await complete_in_thread(record.capture)
                        if reply["status"] != "completed" or stream_error is not None:
                            raise RuntimeError(
                                stream_error or reply.get("error") or f"Proactive agent {reply['status']}",
                            )
                except Exception as exc:
                    stats["model_status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
                    raise
                reminder = None
                if record is not None:
                    async with await get_path_lock(record.path):
                        reminder = await complete_in_thread(self._read_record, record, reply["result"], is_time=is_time)
                self.logger.info(
                    f"{label} result=validated notifications={int(reminder is not None)} "
                    f"elapsed={monotonic() - started:.2f}s path={paths.get('record_path')}",
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
                if (
                    session_id is not None
                    and not asyncio.current_task().cancelling()
                    and asyncio.get_running_loop().time() < deadline
                ):
                    try:
                        async with asyncio.timeout_at(deadline):
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
                stats.update(session_id=session_id, usage=usage)
                update_counts()
        return reminder, session_id
