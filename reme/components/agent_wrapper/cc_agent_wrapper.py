"""Claude Code SDK backend for the unified agent wrapper."""

import json
import shlex
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from .base_agent_wrapper import BaseAgentWrapper
from ..component_registry import R
from ...enumeration import ChunkEnum
from ...schema import AgentReply, StreamChunk, TokenUsage

if TYPE_CHECKING:
    from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage

    from ..job.base_job import BaseJob


@dataclass(frozen=True)
class _BlockState:
    """Metadata needed to correlate one streamed content block."""

    block_id: str | None
    block_type: str
    tool_name: str | None


@R.register("claude_code")
class CcAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by Claude Code SDK."""

    SDK_PACKAGE = "claude-agent-sdk"
    DEFAULT_DISALLOWED_TOOLS = ["WebSearch"]
    MCP_SERVER_NAME = "mcp_server"

    @staticmethod
    def _claude_usage(usage: dict[str, Any] | None) -> TokenUsage:
        """Normalize Claude CLI's input/output usage."""
        return TokenUsage.from_provider(usage or {})

    @property
    def session_path(self) -> Path:
        """Directory used for persisted Claude Code sessions."""
        if self.app_context is None:
            return self.workspace_path / "mem_session"
        return self.workspace_path / self.app_context.app_config.mem_session_dir

    def _ensure_claude_skill_dir(self, config_dir: Path, skills: list[str] | str) -> None:
        """Add selected project skills to Claude Code discovery locations."""
        sources = self._resolve_project_skills(skills)
        if not sources:
            return

        for target in (self.project_path / ".claude" / "skills", config_dir / "skills"):
            try:
                if target.is_symlink():
                    self.logger.warning(f"Preserving existing Claude Code skills link: {target}")
                    continue
                if target.exists() and not target.is_dir():
                    self.logger.warning(f"Preserving existing Claude Code skills path: {target}")
                    continue

                target.mkdir(parents=True, exist_ok=True)
                for skill_name, source in sources.items():
                    skill_target = target / skill_name
                    if skill_target.is_symlink():
                        if skill_target.resolve() != source.resolve():
                            self.logger.warning(f"Preserving existing Claude Code skill link: {skill_target}")
                        continue
                    if skill_target.exists():
                        self.logger.warning(f"Preserving existing Claude Code skill path: {skill_target}")
                        continue
                    skill_target.symlink_to(source, target_is_directory=True)
            except OSError as exc:
                self.logger.warning(f"Failed to link Claude Code skills into {target}: {exc}")

    @classmethod
    def _make_tool(
        cls,
        job: "BaseJob",
        tool_context_id: str | None = None,
        injected_job_kwargs: dict[str, Any] | None = None,
    ):
        from claude_agent_sdk import SdkMcpTool

        injected = cls._resolve_injected_job_kwargs(
            {"tool_context_id": tool_context_id, "injected_job_kwargs": injected_job_kwargs},
        )

        async def run_job(args):
            response = await job(**cls._merge_injected_job_kwargs(dict(args), injected))
            return {
                "content": [{"type": "text", "text": str(response.answer)}],
                "is_error": not response.success,
            }

        return SdkMcpTool(
            name=job.name,
            description=job.description,
            input_schema=cls._strip_injected_parameters(job.parameters, injected),
            handler=run_job,
        )

    def _add_bash_proxy_hook(self, opts: Any) -> None:
        """Inject managed proxy exports into Claude Code Bash commands."""
        proxy_environment = self.command_proxy_environment
        if not proxy_environment:
            return

        from claude_agent_sdk import HookMatcher

        exports = " ".join(f"{name}={shlex.quote(value)}" for name, value in proxy_environment.items())

        async def inject_proxy(hook_input, _tool_use_id, _context):
            tool_input = dict(hook_input["tool_input"])
            command = tool_input.get("command")
            if not isinstance(command, str):
                return {}
            tool_input["command"] = f"export {exports}; {command}"
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "updatedInput": tool_input,
                },
            }

        hooks = dict(opts.hooks or {})
        pre_tool_use = list(hooks.get("PreToolUse") or [])
        pre_tool_use.append(HookMatcher(matcher="Bash", hooks=[inject_proxy]))
        hooks["PreToolUse"] = pre_tool_use
        opts.hooks = hooks

    def _build_options(self, inputs: Any, stream: bool = False, **kwargs) -> Any:
        """Build ClaudeAgentOptions from kwargs.

        ``stream=True`` enables ``include_partial_messages`` so that
        ``StreamEvent`` messages are emitted alongside the final
        ``ResultMessage``.
        """
        from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server

        if not isinstance(inputs, str):
            raise NotImplementedError("Only string input is supported for Claude Code.")

        selected_skills = kwargs.get("skills")
        if isinstance(selected_skills, str) and selected_skills != "all":
            selected_skills = [selected_skills]

        if "setting_sources" not in kwargs and kwargs.get("skills") is None:
            kwargs["setting_sources"] = []
        skip_keys = {"job_tools", "injected_job_kwargs", "output_schema", "api_key", "base_url", "credential"}
        option_fields = {field.name for field in fields(ClaudeAgentOptions)}
        option_kwargs = {key: value for key, value in kwargs.items() if key not in skip_keys and key in option_fields}
        option_kwargs["disallowed_tools"] = list(
            dict.fromkeys(
                [
                    *(kwargs.get("disallowed_tools") or []),
                    *self.DEFAULT_DISALLOWED_TOOLS,
                ],
            ),
        )
        if selected_skills is not None:
            option_kwargs["skills"] = selected_skills
        if stream:
            option_kwargs["include_partial_messages"] = True
        opts = ClaudeAgentOptions(**option_kwargs)

        opts.env = dict(opts.env)
        opts.env.update(self.subprocess_environment)
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url")
        extra_env_dict = {
            "ANTHROPIC_AUTH_TOKEN": api_key if isinstance(api_key, str) else "",
            "ANTHROPIC_BASE_URL": base_url if isinstance(base_url, str) else "",
        }
        if opts.model:
            extra_env_dict.update(
                {
                    "ANTHROPIC_MODEL": opts.model,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": opts.model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": opts.model,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": opts.model,
                },
            )
        opts.env.update(extra_env_dict)
        self._add_bash_proxy_hook(opts)
        self.session_path.mkdir(parents=True, exist_ok=True)
        opts.cwd = opts.cwd or self.cwd
        claude_config_dir = self.session_path / "claude_config"
        opts.env.setdefault("CLAUDE_CONFIG_DIR", str(claude_config_dir))
        if selected_skills is not None:
            self._ensure_claude_skill_dir(claude_config_dir, selected_skills)

        job_tools: list[str] = kwargs.get("job_tools", [])
        resolved_jobs = self._resolve_job_tools(job_tools)
        if resolved_jobs:
            if not isinstance(opts.mcp_servers, dict):
                raise ValueError("job_tools require mcp_servers to be a mapping so the ReMe SDK server can be merged")
            opts.mcp_servers = dict(opts.mcp_servers)
            if self.MCP_SERVER_NAME in opts.mcp_servers:
                raise ValueError(f"mcp_servers already contains reserved server name {self.MCP_SERVER_NAME!r}")
            sdk_tools = [
                self._make_tool(job, kwargs.get("tool_context_id"), kwargs.get("injected_job_kwargs"))
                for job in resolved_jobs
            ]
            opts.mcp_servers[self.MCP_SERVER_NAME] = create_sdk_mcp_server(
                name=self.MCP_SERVER_NAME,
                tools=sdk_tools,
            )
            opts.allowed_tools = list(opts.allowed_tools)
            opts.allowed_tools.extend(job.name for job in resolved_jobs)

        if (output_schema := kwargs.get("output_schema")) is not None:
            opts.output_format = {"type": "json_schema", "schema": output_schema}

        return opts

    # ----- StreamChunk conversion -------------------------------------------

    @classmethod
    # pylint: disable=too-many-return-statements
    def _raw_event_to_chunk(
        cls,
        raw: dict,
        session_id: str | None = None,
        block_states: dict[int, _BlockState] | None = None,
        message_id: str | None = None,
    ) -> StreamChunk:
        """Normalize model-message and block events, retaining unknown payloads."""
        event_type = raw.get("type", "")
        common = {"session_id": session_id, "message_id": message_id, "chunk": None}
        if event_type in {"message_start", "message_stop"}:
            if event_type == "message_start" and block_states is not None:
                block_states.clear()
            return cls._chunk(
                ChunkEnum.DATA,
                **common,
                event="start" if event_type == "message_start" else "end",
                block_id=f"message:{message_id}",
                metadata={"kind": "message", "visibility": "diagnostic", "native": raw},
            )
        if event_type == "message_delta":
            usage = cls._claude_usage(raw.get("usage"))
            common["chunk"] = usage.model_dump()
            return cls._chunk(
                ChunkEnum.USAGE,
                **common,
                event="end",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                metadata={
                    "stop_reason": raw.get("delta", {}).get("stop_reason"),
                    "usage": usage.model_dump(),
                    "native": raw,
                },
            )

        index = raw.get("index", 0)
        block = raw.get("content_block", {})
        if event_type == "content_block_start":
            state = _BlockState(
                block.get("id") or f"{message_id or 'message'}:{index}",
                block.get("type", ""),
                block.get("name"),
            )
            if block_states is not None:
                block_states[index] = state
        else:
            state = block_states.get(index) if block_states is not None else None
        if event_type == "content_block_stop" and block_states is not None:
            block_states.pop(index, None)
        operation = {"content_block_start": "start", "content_block_delta": "delta", "content_block_stop": "end"}.get(
            event_type,
        )
        if state is not None and operation:
            common.update(event=operation, block_id=state.block_id)
            delta = raw.get("delta", {})
            if event_type == "content_block_delta" and delta.get("type") not in {
                "text_delta",
                "thinking_delta",
                "input_json_delta",
            }:
                return cls._chunk(ChunkEnum.DATA, **{**common, "chunk": raw}, metadata={"visibility": "diagnostic"})
            metadata = {
                "native": {key: value for key, value in block.items() if key not in {"text", "thinking", "input"}},
            }
            if state.block_type == "text":
                common["chunk"] = delta.get("text") if operation == "delta" else block.get("text")
                return cls._chunk(ChunkEnum.CONTENT, **common, metadata=metadata)
            if state.block_type == "thinking":
                common["chunk"] = delta.get("thinking") if operation == "delta" else block.get("thinking")
                return cls._chunk(ChunkEnum.THINK, **common, metadata=metadata)
            if state.block_type in {"tool_use", "server_tool_use"}:
                common["chunk"] = delta.get("partial_json") if operation == "delta" else block.get("input") or None
                return cls._chunk(
                    ChunkEnum.TOOL_CALL,
                    **common,
                    tool_call_id=state.block_id,
                    tool_call_name=state.tool_name,
                    metadata=metadata,
                )
        common.update(chunk=raw, event="end")
        return cls._chunk(ChunkEnum.DATA, **common, metadata={"visibility": "diagnostic"})

    @classmethod
    def _message_content_to_chunks(
        cls,
        msg: "AssistantMessage | UserMessage",
        session_id: str | None = None,
        visible_tool_call_ids: set[str] | None = None,
        include_text: bool = False,
        message_id: str | None = None,
    ) -> list[StreamChunk]:
        """Use completed SDK blocks as authoritative snapshots, including tool results."""
        from claude_agent_sdk import (
            ServerToolResultBlock,
            ServerToolUseBlock,
            TextBlock,
            ThinkingBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        chunks: list[StreamChunk] = []
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            if include_text:
                chunks.append(
                    cls._chunk(
                        ChunkEnum.CONTENT,
                        session_id=session_id,
                        message_id=message_id,
                        block_id=f"{message_id}:0",
                        event="end",
                        chunk=content,
                    ),
                )
            return chunks
        for index, block in enumerate(content or []):
            common = {
                "session_id": session_id,
                "message_id": message_id,
                "event": "end",
                "block_id": f"{message_id or 'message'}:{index}",
            }
            if include_text and isinstance(block, (TextBlock, ThinkingBlock)):
                thinking = isinstance(block, ThinkingBlock)
                chunks.append(
                    cls._chunk(
                        ChunkEnum.THINK if thinking else ChunkEnum.CONTENT,
                        **common,
                        chunk=block.thinking if thinking else block.text,
                        metadata={"native": {"signature": block.signature}} if thinking else {},
                    ),
                )
            elif include_text and isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
                common["block_id"] = block.id
                chunks.append(
                    cls._chunk(
                        ChunkEnum.TOOL_CALL,
                        **common,
                        tool_call_id=block.id,
                        tool_call_name=block.name,
                        chunk=block.input,
                    ),
                )
                if visible_tool_call_ids is not None:
                    visible_tool_call_ids.add(block.id)
            elif isinstance(block, (ToolResultBlock, ServerToolResultBlock)):
                if visible_tool_call_ids is not None and block.tool_use_id not in visible_tool_call_ids:
                    continue
                common["block_id"] = block.tool_use_id
                payload = {"tool_use_id": block.tool_use_id, "content": block.content}
                if isinstance(block, ToolResultBlock):
                    payload["is_error"] = block.is_error
                chunks.append(
                    cls._chunk(
                        ChunkEnum.TOOL_RESULT,
                        **common,
                        tool_call_id=block.tool_use_id,
                        chunk=payload,
                        status="failed" if payload.get("is_error") else "completed",
                    ),
                )
        return chunks

    @classmethod
    def _result_error_text(cls, msg: "ResultMessage") -> str:
        """Return the error text used by both the SDK and unified chunks."""
        return "; ".join(msg.errors or []) or msg.result or str(msg.subtype)

    @classmethod
    def _result_message_to_chunks(cls, msg: "ResultMessage") -> list[StreamChunk]:
        """Convert the SDK terminal result into usage and error chunks."""
        session_id = msg.session_id or ""
        usage = msg.usage or {}
        normalized = cls._claude_usage(usage)
        chunks = [
            cls._chunk(
                ChunkEnum.USAGE,
                session_id=session_id,
                chunk=json.dumps(normalized.model_dump()),
                input_tokens=normalized.input_tokens,
                output_tokens=normalized.output_tokens,
                metadata={
                    "usage": normalized.model_dump(),
                    "duration_ms": msg.duration_ms,
                    "duration_api_ms": msg.duration_api_ms,
                    "stop_reason": msg.stop_reason,
                    "num_turns": msg.num_turns,
                    "total_cost_usd": msg.total_cost_usd,
                    "model_usage": msg.model_usage,
                    "permission_denials": msg.permission_denials,
                    "deferred_tool_use": (asdict(msg.deferred_tool_use) if msg.deferred_tool_use else None),
                    "api_error_status": msg.api_error_status,
                    "native": cls._serialize(msg),
                },
            ),
        ]
        if msg.is_error:
            chunks.append(
                cls._chunk(
                    ChunkEnum.ERROR,
                    session_id=session_id,
                    chunk=cls._result_error_text(msg),
                    metadata={"api_error_status": msg.api_error_status},
                ),
            )
        return chunks

    # ----- reply / reply_stream --------------------------------------------

    async def compact_session(self, session_id: str) -> None:
        """Compact a Claude Code session through its native command."""
        result = await self.reply("/compact", resume=session_id)
        if result["status"] != "completed":
            raise RuntimeError(result["error"] or f"Claude Code session compaction {result['status']}")

    async def reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream one invocation, keeping model-message boundaries inside it."""
        from claude_agent_sdk import (
            AssistantMessage,
            MirrorErrorMessage,
            query,
            RateLimitEvent,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            UserMessage,
        )

        kwargs = self._merged_kwargs(kwargs)
        opts = self._build_options(inputs, stream=True, **kwargs)
        block_states: dict[str | None, dict[int, _BlockState]] = {}
        visible_tool_call_ids: set[str] = set()
        turn_id = uuid4().hex
        partial_message_ids: dict[str | None, str] = {}
        current_session_id = None
        emitted_content = False
        started = False
        result: AgentReply | None = None
        usage = None
        expected_trailing_error = None
        try:
            async with aclosing(query(prompt=inputs, options=opts)) as stream:
                async for msg in stream:
                    current_session_id = getattr(msg, "session_id", None) or current_session_id
                    if not started:
                        started = True
                        yield self._chunk(
                            ChunkEnum.REPLY_START,
                            event="start",
                            turn_id=turn_id,
                            session_id=current_session_id,
                            status="running",
                        )
                    if expected_trailing_error is not None and not (
                        isinstance(msg, SystemMessage) and msg.subtype == "session_state_changed"
                    ):
                        expected_trailing_error = None
                    chunks = []
                    parent_id = getattr(msg, "parent_tool_use_id", None)
                    if isinstance(msg, StreamEvent):
                        if msg.event.get("type") == "message_start":
                            partial_message_ids[parent_id] = msg.event.get("message", {}).get("id") or msg.uuid
                        chunk = self._raw_event_to_chunk(
                            msg.event,
                            session_id=current_session_id,
                            block_states=block_states.setdefault(parent_id, {}),
                            message_id=partial_message_ids.get(parent_id),
                        )
                        chunk.metadata["parent_tool_use_id"] = msg.parent_tool_use_id
                        if chunk.chunk_type == ChunkEnum.TOOL_CALL and chunk.tool_call_id:
                            visible_tool_call_ids.add(chunk.tool_call_id)
                        chunks.append(chunk)
                    elif isinstance(msg, (AssistantMessage, UserMessage)):
                        message_id = (
                            getattr(msg, "message_id", None)
                            or partial_message_ids.get(parent_id)
                            or getattr(msg, "uuid", None)
                            or uuid4().hex
                        )
                        chunks = self._message_content_to_chunks(
                            msg,
                            current_session_id,
                            visible_tool_call_ids,
                            include_text=isinstance(msg, AssistantMessage),
                            message_id=message_id,
                        )
                        if isinstance(msg, AssistantMessage):
                            partial_message_ids.pop(parent_id, None)
                    elif isinstance(msg, ResultMessage):
                        if msg.is_error:
                            expected_trailing_error = (
                                f"Claude Code returned an error result: {self._result_error_text(msg)}"
                            )
                        if not emitted_content and msg.result:
                            chunks.append(
                                self._chunk(
                                    ChunkEnum.CONTENT,
                                    event="end",
                                    block_id=f"{turn_id}:result",
                                    chunk=msg.result,
                                ),
                            )
                        chunks.extend(self._result_message_to_chunks(msg))
                        status = "completed"
                        if msg.terminal_reason in {"aborted_streaming", "aborted_tools"}:
                            status = "interrupted"
                        elif msg.is_error or msg.subtype != "success":
                            status = "failed"
                        elif msg.deferred_tool_use is not None:
                            status = "waiting"
                        usage = self._claude_usage(msg.usage) if msg.usage is not None else None
                        result = self._reply_result(
                            session_id=msg.session_id or "",
                            result=msg.result,
                            structured_output=msg.structured_output,
                            usage=usage,
                            status=status,
                            error=self._result_error_text(msg) if status == "failed" else None,
                            raw=self._serialize(msg),
                        )
                    elif isinstance(msg, MirrorErrorMessage):
                        chunks.append(
                            self._chunk(
                                ChunkEnum.DATA,
                                event="end",
                                chunk=f"Session mirror failed: {msg.error}",
                                metadata={"event": "session_mirror_error", "session_key": msg.key},
                            ),
                        )
                    elif isinstance(msg, RateLimitEvent):
                        if msg.rate_limit_info.status == "rejected":
                            chunks.append(self._chunk(ChunkEnum.ERROR, chunk="Rate limit exceeded"))
                    else:
                        chunks.append(
                            self._chunk(
                                ChunkEnum.DATA,
                                event="end",
                                chunk=self._serialize(msg),
                                metadata={"visibility": "diagnostic"},
                            ),
                        )
                    if not chunks:
                        chunks = [self._chunk(ChunkEnum.DATA, chunk=None, metadata={"visibility": "diagnostic"})]
                    for chunk in chunks:
                        chunk.metadata.pop("native", None)
                    chunks[0].metadata["native"] = self._serialize(msg)
                    for chunk in chunks:
                        chunk.metadata["parent_tool_use_id"] = parent_id
                        chunk.turn_id = turn_id
                        chunk.session_id = chunk.session_id or current_session_id
                        if chunk.chunk_type == ChunkEnum.CONTENT and chunk.chunk:
                            emitted_content = True
                        yield chunk
        except Exception as exc:
            if expected_trailing_error is None or str(exc) != expected_trailing_error:
                raise
            self.logger.debug(f"Ignoring Claude Code process exit after error result: {exc}")
        finally:
            if usage is not None:
                self._record_token_usage(usage)
        if result is None:
            raise RuntimeError("Claude Code result message not received")
        yield self._chunk(
            ChunkEnum.REPLY_END,
            event="end",
            session_id=result["session_id"],
            turn_id=turn_id,
            status=result["status"],
            chunk={key: value for key, value in result.items() if key != "events"},
            metadata={"error": result["error"]},
        )
