"""Claude Code SDK backend for the unified agent wrapper."""

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .base_agent_wrapper import BaseAgentWrapper
from ..component_registry import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk

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
    def _make_tool(cls, job: "BaseJob", tool_context_id: str | None = None):
        from claude_agent_sdk import SdkMcpTool

        async def run_job(args):
            call_args = dict(args)
            if tool_context_id:
                if "tool_context_id" in call_args:
                    raise ValueError("tool_context_id is injected by agent_wrapper")
                call_args["tool_context_id"] = tool_context_id
            response = await job(**call_args)
            return {
                "content": [{"type": "text", "text": str(response.answer)}],
                "is_error": not response.success,
            }

        return SdkMcpTool(
            name=job.name,
            description=job.description,
            input_schema=job.parameters,
            handler=run_job,
        )

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
        skip_keys = {"job_tools", "output_schema", "api_key", "base_url", "credential"}
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
            sdk_tools = [self._make_tool(job, kwargs.get("tool_context_id")) for job in resolved_jobs]
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
    ) -> StreamChunk | None:
        """Convert a raw Anthropic streaming event dict to a StreamChunk.

        ``block_states`` maps each content-block index to metadata captured at
        ``content_block_start`` for use by later delta and stop events.

        Returns ``None`` for events that should be silently skipped.
        """
        event_type = raw.get("type")

        # --- Message-level lifecycle ----------------------------------------

        if event_type == "message_start":
            if block_states is not None:
                block_states.clear()
            message = raw.get("message", {})
            meta = {
                "message_id": message.get("id"),
                "model": message.get("model"),
                "role": message.get("role"),
            }
            return cls._chunk(ChunkEnum.REPLY_START, session_id=session_id, chunk="", metadata=meta)

        if event_type == "message_delta":
            delta = raw.get("delta", {})
            usage = raw.get("usage", {})
            return cls._chunk(
                ChunkEnum.USAGE,
                session_id=session_id,
                chunk=json.dumps(usage),
                output_tokens=usage.get("output_tokens"),
                metadata={"stop_reason": delta.get("stop_reason")},
            )

        if event_type == "message_stop":
            return cls._chunk(ChunkEnum.REPLY_END, session_id=session_id, chunk="")

        # --- Content-block lifecycle ----------------------------------------

        if event_type == "content_block_start":
            idx, content_block = raw.get("index", 0), raw.get("content_block", {})
            block_type, bid = content_block.get("type", ""), content_block.get("id", "")
            if block_states is not None:
                block_states[idx] = _BlockState(bid or None, block_type, content_block.get("name"))

            if block_type == "text":
                return cls._chunk(ChunkEnum.CONTENT, block_id=bid, chunk=content_block.get("text", ""))
            if block_type == "thinking":
                return cls._chunk(
                    ChunkEnum.THINK,
                    block_id=bid,
                    chunk=content_block.get("thinking", ""),
                )
            if block_type in {"tool_use", "server_tool_use"}:
                payload = {
                    "name": content_block.get("name"),
                    "id": content_block.get("id"),
                }
                return cls._chunk(
                    ChunkEnum.TOOL_CALL,
                    block_id=bid,
                    tool_call_id=content_block.get("id"),
                    tool_call_name=content_block.get("name"),
                    chunk=json.dumps(payload),
                )
            return None

        if event_type == "content_block_delta":
            delta = raw.get("delta", {})
            delta_type = delta.get("type", "")
            idx = raw.get("index", 0)
            state = block_states.get(idx) if block_states else None
            bid = state.block_id if state else None
            tool_name = state.tool_name if state else None

            if delta_type == "text_delta":
                return cls._chunk(ChunkEnum.CONTENT, block_id=bid, chunk=delta.get("text", ""))
            if delta_type == "thinking_delta":
                return cls._chunk(ChunkEnum.THINK, block_id=bid, chunk=delta.get("thinking", ""))
            if delta_type == "input_json_delta":
                return cls._chunk(
                    ChunkEnum.TOOL_CALL,
                    block_id=bid,
                    tool_call_id=bid,
                    tool_call_name=tool_name,
                    chunk=delta.get("partial_json", ""),
                )
            return None

        if event_type == "content_block_stop":
            idx = raw.get("index", 0)
            state = block_states.pop(idx, None) if block_states else None
            bid = state.block_id if state else None
            block_type = state.block_type if state else None
            tool_name = state.tool_name if state else None

            if block_type in {"tool_use", "server_tool_use"}:
                return cls._chunk(
                    ChunkEnum.TOOL_CALL,
                    block_id=bid,
                    tool_call_id=bid,
                    tool_call_name=tool_name,
                    chunk="",
                )
            if block_type == "thinking":
                return cls._chunk(ChunkEnum.THINK, block_id=bid, chunk="")
            # text or unknown -> CONTENT
            return cls._chunk(ChunkEnum.CONTENT, block_id=bid, chunk="")

        # Ping / other unknown types -> skip
        return None

    @classmethod
    def _message_content_to_chunks(
        cls,
        msg: "AssistantMessage | UserMessage",
        session_id: str | None = None,
        visible_tool_call_ids: set[str] | None = None,
        include_text: bool = False,
    ) -> list[StreamChunk]:
        """Convert typed SDK content blocks that are not partial events."""
        from claude_agent_sdk import ServerToolResultBlock, TextBlock, ToolResultBlock

        chunks: list[StreamChunk] = []
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            if include_text and content:
                chunks.append(cls._chunk(ChunkEnum.CONTENT, session_id=session_id, chunk=content))
            return chunks
        if content is None:
            return chunks

        for block in content:
            if include_text and isinstance(block, TextBlock) and block.text:
                chunks.append(cls._chunk(ChunkEnum.CONTENT, session_id=session_id, chunk=block.text))
            elif isinstance(block, (ToolResultBlock, ServerToolResultBlock)):
                tool_use_id = block.tool_use_id
                if visible_tool_call_ids is not None and tool_use_id not in visible_tool_call_ids:
                    continue
                payload: dict[str, Any] = {
                    "tool_use_id": tool_use_id,
                    "content": block.content,
                }
                if isinstance(block, ToolResultBlock):
                    payload["is_error"] = block.is_error
                chunks.append(
                    cls._chunk(
                        ChunkEnum.TOOL_RESULT,
                        session_id=session_id,
                        block_id=tool_use_id,
                        tool_call_id=tool_use_id,
                        chunk=payload,
                    ),
                )

        return chunks

    @classmethod
    def _result_error_text(cls, msg: "ResultMessage") -> str:
        """Return the error text used by both the SDK and unified chunks."""
        return "; ".join(msg.errors or []) or str(msg.subtype)

    @classmethod
    def _result_message_to_chunks(cls, msg: "ResultMessage") -> list[StreamChunk]:
        """Convert the SDK terminal result into usage and error chunks."""
        session_id = msg.session_id or ""
        usage = msg.usage or {}
        chunks = [
            cls._chunk(
                ChunkEnum.USAGE,
                session_id=session_id,
                chunk=json.dumps(usage),
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                metadata={
                    "duration_ms": msg.duration_ms,
                    "duration_api_ms": msg.duration_api_ms,
                    "stop_reason": msg.stop_reason,
                    "num_turns": msg.num_turns,
                    "total_cost_usd": msg.total_cost_usd,
                    "model_usage": msg.model_usage,
                    "permission_denials": msg.permission_denials,
                    "deferred_tool_use": (asdict(msg.deferred_tool_use) if msg.deferred_tool_use else None),
                    "api_error_status": msg.api_error_status,
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
        if result["last_message"].get("is_error"):
            raise RuntimeError("Claude Code session compaction failed")

    async def reply(self, inputs: Any, **kwargs) -> dict:
        from claude_agent_sdk import query, ResultMessage

        kwargs = self._merged_kwargs(kwargs)
        opts = self._build_options(inputs, stream=False, **kwargs)

        last_msg = None
        async with aclosing(query(prompt=inputs, options=opts)) as stream:
            async for msg in stream:
                if isinstance(msg, ResultMessage):
                    last_msg = msg

        if last_msg is None:
            raise ValueError("No message received from Claude Code.")

        result = {
            "session_id": last_msg.session_id or "",
            "last_message": asdict(last_msg),
            "result": last_msg.result,
        }
        if kwargs.get("output_schema") is not None:
            result["structured_output"] = last_msg.structured_output
        return result

    async def reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream Claude Code events as unified StreamChunk objects."""
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

        kwargs = self._merged_stream_kwargs(kwargs)
        opts = self._build_options(inputs, stream=True, **kwargs)

        block_states: dict[int, _BlockState] = {}
        visible_tool_call_ids: set[str] = set()
        current_session_id: str | None = None
        emitted_content = False
        emitted_reply_end = False
        reply_open = False
        expected_trailing_error: str | None = None

        try:
            async with aclosing(query(prompt=inputs, options=opts)) as stream:
                async for msg in stream:
                    if expected_trailing_error is not None and not (
                        isinstance(msg, SystemMessage) and msg.subtype == "session_state_changed"
                    ):
                        expected_trailing_error = None

                    if isinstance(msg, StreamEvent):
                        current_session_id = msg.session_id or current_session_id
                        chunk = self._raw_event_to_chunk(
                            msg.event,
                            session_id=msg.session_id,
                            block_states=block_states,
                        )
                        if chunk is not None:
                            chunk.session_id = chunk.session_id or msg.session_id
                            if chunk.chunk_type == ChunkEnum.TOOL_CALL and chunk.tool_call_id:
                                visible_tool_call_ids.add(chunk.tool_call_id)
                            if chunk.chunk_type == ChunkEnum.CONTENT and chunk.chunk:
                                emitted_content = True
                            if chunk.chunk_type == ChunkEnum.REPLY_START:
                                reply_open = True
                            if chunk.chunk_type == ChunkEnum.REPLY_END:
                                emitted_reply_end = True
                                reply_open = False
                            yield chunk

                    elif isinstance(msg, UserMessage):
                        for chunk in self._message_content_to_chunks(msg, current_session_id, visible_tool_call_ids):
                            yield chunk

                    elif isinstance(msg, ResultMessage):
                        if msg.is_error:
                            expected_trailing_error = (
                                f"Claude Code returned an error result: {self._result_error_text(msg)}"
                            )
                        current_session_id = msg.session_id or current_session_id
                        if not emitted_content and msg.result:
                            emitted_content = True
                            yield self._chunk(
                                ChunkEnum.CONTENT,
                                session_id=msg.session_id or "",
                                chunk=msg.result,
                            )
                        for chunk in self._result_message_to_chunks(msg):
                            yield chunk
                        if reply_open or not emitted_reply_end:
                            emitted_reply_end = True
                            reply_open = False
                            yield self._chunk(
                                ChunkEnum.REPLY_END,
                                session_id=current_session_id,
                                chunk="",
                            )

                    elif isinstance(msg, AssistantMessage):
                        current_session_id = msg.session_id or current_session_id
                        for chunk in self._message_content_to_chunks(
                            msg,
                            current_session_id,
                            visible_tool_call_ids,
                            include_text=not emitted_content,
                        ):
                            if chunk.chunk_type == ChunkEnum.CONTENT and chunk.chunk:
                                emitted_content = True
                            yield chunk

                    elif isinstance(msg, MirrorErrorMessage):
                        self.logger.warning(f"Claude Code session mirror failed: {msg.error}")
                        yield self._chunk(
                            ChunkEnum.DATA,
                            session_id=current_session_id,
                            chunk=f"Session mirror failed: {msg.error}",
                            metadata={
                                "event": "session_mirror_error",
                                "session_key": msg.key,
                            },
                        )

                    elif isinstance(msg, RateLimitEvent) and msg.rate_limit_info.status == "rejected":
                        yield self._chunk(
                            ChunkEnum.ERROR,
                            session_id=msg.session_id,
                            chunk="Rate limit exceeded",
                        )
        except Exception as exc:
            if expected_trailing_error is None or str(exc) != expected_trailing_error:
                raise
            self.logger.debug(f"Ignoring Claude Code process exit after error result: {exc}")
