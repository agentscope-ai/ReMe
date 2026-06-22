"""Claude Code SDK backend for the unified agent wrapper."""

import json
import os
import shutil
from collections.abc import AsyncGenerator
from dataclasses import asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .base_agent_wrapper import BaseAgentWrapper
from ..component_registry import R
from ...enumeration import ChunkEnum
from ...schema import StreamChunk
from ...utils.env_utils import load_env

if TYPE_CHECKING:
    from ..job.base_job import BaseJob
    from claude_agent_sdk.types import SessionKey, SessionStoreEntry, SessionStoreListEntry


class _CcFileSessionStore:
    """File-backed Claude Code SessionStore rooted under the ReMe workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def _safe_parts(value: str) -> list[str]:
        parts = [part for part in value.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"Invalid session store path component: {value!r}")
        return parts

    def _path_for_key(self, key: "SessionKey") -> Path:
        session_id = key["session_id"]
        subpath = key.get("subpath")

        path = self.root.joinpath(*self._safe_parts(session_id))
        if subpath:
            path = path.joinpath(*self._safe_parts(subpath))
        else:
            path = path.with_suffix(".jsonl")
        if subpath:
            path = path.with_suffix(".jsonl")

        resolved_root = self.root.resolve()
        resolved_path = path.resolve()
        if resolved_root != resolved_path and resolved_root not in resolved_path.parents:
            raise ValueError(f"Session store path escapes root: {resolved_path}")
        return path

    @staticmethod
    def _read_entries(path: Path) -> list["SessionStoreEntry"]:
        """Read JSONL session-store entries from disk."""
        if not path.exists():
            return []
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
        return entries

    async def append(self, key: "SessionKey", entries: list["SessionStoreEntry"]) -> None:
        """Append new session-store entries, deduplicating by UUID."""
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing_uuids = {
            entry.get("uuid") for entry in self._read_entries(path) if isinstance(entry, dict) and entry.get("uuid")
        }
        new_entries = [
            entry for entry in entries if not (isinstance(entry, dict) and entry.get("uuid") in existing_uuids)
        ]
        if not new_entries:
            return

        with path.open("a", encoding="utf-8") as f:
            for entry in new_entries:
                f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")

    async def load(self, key: "SessionKey") -> list["SessionStoreEntry"] | None:
        """Load session-store entries for a key."""
        path = self._path_for_key(key)
        if not path.exists():
            return None
        return self._read_entries(path)

    async def list_sessions(self, _project_key: str) -> list["SessionStoreListEntry"]:
        """List root-level Claude Code sessions."""
        if not self.root.exists():
            return []
        return [
            {"session_id": path.stem, "mtime": int(path.stat().st_mtime * 1000)}
            for path in self.root.glob("*.jsonl")
            if path.is_file()
        ]

    async def delete(self, key: "SessionKey") -> None:
        """Delete a session-store entry and any subkey directory."""
        path = self._path_for_key(key)
        if path.exists():
            path.unlink()

        if not key.get("subpath"):
            session_dir = self.root.joinpath(*self._safe_parts(key["session_id"]))
            if session_dir.exists():
                for child in sorted(session_dir.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                session_dir.rmdir()

    async def list_subkeys(self, key: dict[str, str]) -> list[str]:
        """List subkeys below a root session key."""
        session_dir = self.root.joinpath(*self._safe_parts(key["session_id"]))
        if not session_dir.exists():
            return []
        subkeys = []
        for path in session_dir.rglob("*.jsonl"):
            if path.is_file():
                subkeys.append(str(path.relative_to(session_dir).with_suffix("")))
        return subkeys


@R.register("claude_code")
class CcAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by Claude Code SDK."""

    DEFAULT_DISALLOWED_TOOLS = ["WebSearch"]

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        for value in values:
            if isinstance(value, str) and value:
                return value
        return ""

    def _default_llm_credential(self) -> dict[str, Any]:
        """Return the default as_llm credential config, if available."""
        if self.app_context is None:
            return {}
        components = self.app_context.app_config.components
        llm_configs = components.get("as_llm") or components.get("AS_LLM") or components.get("as_llm".upper())
        if llm_configs is None:
            from ...enumeration import ComponentEnum

            llm_configs = components.get(ComponentEnum.AS_LLM)
        if not isinstance(llm_configs, dict):
            return {}

        default_llm = llm_configs.get("default")
        credential = getattr(default_llm, "credential", None)
        return credential if isinstance(credential, dict) else {}

    def _claude_code_api_env(self, kwargs: dict[str, Any]) -> dict[str, str]:
        """Resolve Anthropic-compatible API environment for Claude Code."""
        credential = kwargs.get("credential") if isinstance(kwargs.get("credential"), dict) else {}
        default_credential = self._default_llm_credential()

        base_url = self._first_non_empty(
            kwargs.get("base_url"),
            credential.get("base_url"),
            os.getenv("ANTHROPIC_BASE_URL"),
            os.getenv("CLAUDE_CODE_BASE_URL"),
            os.getenv("LLM_BASE_URL"),
            default_credential.get("base_url"),
        )
        api_key = self._first_non_empty(
            kwargs.get("api_key"),
            credential.get("api_key"),
            os.getenv("ANTHROPIC_AUTH_TOKEN"),
            os.getenv("CLAUDE_CODE_API_KEY"),
            os.getenv("LLM_API_KEY"),
            default_credential.get("api_key"),
        )

        env: dict[str, str] = {}
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
        if api_key:
            env["ANTHROPIC_AUTH_TOKEN"] = api_key
        return env

    @property
    def session_path(self) -> Path:
        """Directory used for persisted Claude Code sessions."""
        if self.app_context is None:
            return self.workspace_path / "session"
        return self.workspace_path / self.app_context.app_config.session_dir

    def _ensure_claude_skill_dir(self, config_dir: Path) -> None:
        """Expose project skills through Claude Code skill discovery locations."""
        project_skills = self.project_skills_root
        if not project_skills.exists():
            return

        for target in (self.project_path / ".claude" / "skills", config_dir / "skills"):
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if target.exists() or target.is_symlink():
                    if target.resolve() == project_skills.resolve():
                        continue
                    if target.is_dir() and not target.is_symlink():
                        shutil.rmtree(target)
                    else:
                        target.unlink()

                target.symlink_to(project_skills, target_is_directory=True)
            except OSError as exc:
                self.logger.warning(f"Failed to link Claude Code skills directory {target}: {exc}")

    @staticmethod
    def _make_tool(job: "BaseJob"):
        from claude_agent_sdk import SdkMcpTool

        async def run_job(args):
            response = await job(**args)
            return {"content": [{"type": "text", "text": str(response.answer)}], "is_error": not response.success}

        return SdkMcpTool(name=job.name, description=job.description, input_schema=job.parameters, handler=run_job)

    def _build_options(self, inputs: Any, stream: bool = False, **kwargs) -> Any:
        """Build ClaudeAgentOptions from kwargs.

        ``stream=True`` enables ``include_partial_messages`` so that
        ``StreamEvent`` messages are emitted alongside the final
        ``ResultMessage``.
        """
        from claude_agent_sdk import create_sdk_mcp_server
        from claude_agent_sdk.types import ClaudeAgentOptions

        kwargs = self._merged_kwargs(kwargs)

        skills = kwargs.get("skills")
        if isinstance(skills, str) and skills != "all":
            kwargs["skills"] = [skills]

        if "setting_sources" not in kwargs and kwargs.get("skills") is None:
            kwargs["setting_sources"] = []
        disallowed_tools = list(kwargs.get("disallowed_tools") or [])
        for tool_name in self.DEFAULT_DISALLOWED_TOOLS:
            if tool_name not in disallowed_tools:
                disallowed_tools.append(tool_name)
        kwargs["disallowed_tools"] = disallowed_tools

        opts = ClaudeAgentOptions()
        if stream:
            opts.include_partial_messages = True

        skip_keys = {"job_tools", "output_schema", "api_key", "base_url", "credential"}
        for k, v in kwargs.items():
            if k not in skip_keys and hasattr(opts, k):
                setattr(opts, k, v)

        model = getattr(opts, "model", None) or kwargs.get("model")
        project_env = self.project_path / ".env"
        opts.env.update(load_env(project_env) if project_env.exists() else load_env())
        extra_env_dict: dict = self._claude_code_api_env(kwargs)
        if model:
            extra_env_dict.update(
                {
                    "ANTHROPIC_MODEL": model,
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
                    "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                    "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
                },
            )
        opts.env.update(extra_env_dict)
        self.session_path.mkdir(parents=True, exist_ok=True)
        opts.cwd = opts.cwd or self.project_path
        claude_config_dir = self.session_path / "claude_config"
        opts.env.setdefault("CLAUDE_CONFIG_DIR", str(claude_config_dir))
        if opts.skills is not None:
            self._ensure_claude_skill_dir(claude_config_dir)
        opts.session_store = opts.session_store or _CcFileSessionStore(self.session_path / "claude_code")

        job_tools: list[str] = kwargs.get("job_tools", [])
        resolved_jobs = self._resolve_job_tools(job_tools)
        if resolved_jobs:
            sdk_tools = [self._make_tool(job) for job in resolved_jobs]
            server = create_sdk_mcp_server(name="mcp_server", tools=sdk_tools)
            opts.mcp_servers = opts.mcp_servers if isinstance(opts.mcp_servers, dict) else {}
            opts.mcp_servers["mcp_server"] = server
            opts.allowed_tools.extend(job.name for job in resolved_jobs)

        if output_schema := kwargs.get("output_schema"):
            opts.output_format = {"type": "json_schema", "schema": output_schema}

        if not isinstance(inputs, str):
            raise NotImplementedError("Only string input is supported for Claude Code.")

        return opts

    # ----- StreamChunk conversion -------------------------------------------

    @classmethod
    # pylint: disable=too-many-return-statements
    def _raw_event_to_chunk(
        cls,
        raw: dict,
        session_id: str | None = None,
        block_ids: dict[int, str] | None = None,
        block_types: dict[int, str] | None = None,
        tool_call_names: dict[int, str] | None = None,
    ) -> StreamChunk | None:
        """Convert a raw Anthropic streaming event dict to a StreamChunk.

        ``block_ids`` / ``block_types`` / ``tool_call_names`` map
        content-block ``index`` to metadata tracked from the
        ``content_block_start`` event, so that later delta / stop
        events can reference the correct ``block_id`` and
        ``chunk_type``.

        Returns ``None`` for events that should be silently skipped.
        """
        event_type = raw.get("type")

        # --- Message-level lifecycle ----------------------------------------

        if event_type == "message_start":
            message = raw.get("message", {})
            meta = {"message_id": message.get("id"), "model": message.get("model"), "role": message.get("role")}
            return cls._chunk(ChunkEnum.REPLY_START, session_id=session_id, chunk="", metadata=meta)

        if event_type == "message_delta":
            delta = raw.get("delta", {})
            usage = raw.get("usage", {})
            return cls._chunk(
                ChunkEnum.REPLY_END,
                session_id=session_id,
                chunk="",
                output_tokens=usage.get("output_tokens"),
                metadata={"stop_reason": delta.get("stop_reason")},
            )

        if event_type == "message_stop":
            return cls._chunk(ChunkEnum.REPLY_END, session_id=session_id, chunk="")

        # --- Content-block lifecycle ----------------------------------------

        if event_type == "content_block_start":
            idx, content_block = raw.get("index", 0), raw.get("content_block", {})
            block_type, bid = content_block.get("type", ""), content_block.get("id", "")

            # Track for later delta / stop correlation
            if block_ids is not None and bid:
                block_ids[idx] = bid
            if block_types is not None and block_type:
                block_types[idx] = block_type
            if tool_call_names is not None and content_block.get("name"):
                tool_call_names[idx] = content_block["name"]

            if block_type == "text":
                return cls._chunk(ChunkEnum.CONTENT, block_id=bid, chunk=content_block.get("text", ""))
            if block_type == "thinking":
                return cls._chunk(ChunkEnum.THINK, block_id=bid, chunk=content_block.get("thinking", ""))
            if block_type == "tool_use":
                payload = {"name": content_block.get("name"), "id": content_block.get("id")}
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
            bid = block_ids.get(idx) if block_ids else None
            tc_name = tool_call_names.get(idx) if tool_call_names else None

            if delta_type == "text_delta":
                return cls._chunk(ChunkEnum.CONTENT, block_id=bid, chunk=delta.get("text", ""))
            if delta_type == "thinking_delta":
                return cls._chunk(ChunkEnum.THINK, block_id=bid, chunk=delta.get("thinking", ""))
            if delta_type == "input_json_delta":
                return cls._chunk(
                    ChunkEnum.TOOL_CALL,
                    block_id=bid,
                    tool_call_id=bid,
                    tool_call_name=tc_name,
                    chunk=delta.get("partial_json", ""),
                )
            return None

        if event_type == "content_block_stop":
            idx = raw.get("index", 0)
            bid = block_ids.get(idx) if block_ids else None
            btype = block_types.get(idx) if block_types else None
            tc_name = tool_call_names.get(idx) if tool_call_names else None

            if btype == "tool_use":
                return cls._chunk(ChunkEnum.TOOL_CALL, block_id=bid, tool_call_id=bid, tool_call_name=tc_name, chunk="")
            if btype == "thinking":
                return cls._chunk(ChunkEnum.THINK, block_id=bid, chunk="")
            # text or unknown -> CONTENT
            return cls._chunk(ChunkEnum.CONTENT, block_id=bid, chunk="")

        # Ping / other unknown types -> skip
        return None

    @classmethod
    def _message_content_to_chunks(
        cls,
        msg: Any,
        session_id: str | None = None,
        visible_tool_call_ids: set[str] | None = None,
        include_text: bool = False,
    ) -> list[StreamChunk]:
        """Convert non-partial SDK message content blocks into stream chunks.

        Claude Code streams assistant text/tool-use deltas as ``StreamEvent``
        objects, but tool results can arrive later as regular message content
        blocks.  Surface those blocks so the UI can show what each tool
        returned.  Some SDK/CLI combinations also put assistant text only in
        regular message blocks, so callers can opt into text conversion.
        """
        chunks: list[StreamChunk] = []
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            return chunks

        for block in content:
            block_name = block.__class__.__name__
            if include_text and block_name == "TextBlock":
                text = getattr(block, "text", "")
                if text:
                    chunks.append(cls._chunk(ChunkEnum.CONTENT, session_id=session_id, chunk=text))
            elif include_text and isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    chunks.append(cls._chunk(ChunkEnum.CONTENT, session_id=session_id, chunk=text))
            elif include_text and isinstance(block, str):
                chunks.append(cls._chunk(ChunkEnum.CONTENT, session_id=session_id, chunk=block))
            elif block_name in {"ToolResultBlock", "ServerToolResultBlock"}:
                tool_use_id = getattr(block, "tool_use_id", None)
                if visible_tool_call_ids is not None and tool_use_id not in visible_tool_call_ids:
                    continue
                payload: dict[str, Any] = {
                    "tool_use_id": tool_use_id,
                    "content": getattr(block, "content", None),
                }
                if hasattr(block, "is_error"):
                    payload["is_error"] = getattr(block, "is_error")
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

    @staticmethod
    def _result_message_is_error(msg: Any) -> bool:
        """Return whether an SDK ResultMessage represents a failed result."""
        subtype = getattr(msg, "subtype", None)
        if isinstance(subtype, str) and subtype.lower() == "success":
            return False

        is_error = getattr(msg, "is_error", False)
        if isinstance(is_error, bool):
            return is_error
        if isinstance(is_error, str):
            return is_error.lower() in {"true", "error", "errored", "failed", "failure"}

        return isinstance(subtype, str) and subtype.lower() in {"error", "failed", "failure"}

    @staticmethod
    def _is_trailing_success_error(exc: Exception) -> bool:
        """Return whether an SDK iterator error is the known success-exit artifact."""
        return "Claude Code returned an error result: success" in str(exc)

    # ----- reply / reply_stream --------------------------------------------

    async def reply(self, inputs: Any, **kwargs) -> dict:
        from claude_agent_sdk import query, ResultMessage

        opts = self._build_options(inputs, stream=False, **kwargs)

        last_msg = None
        async for msg in query(prompt=inputs, options=opts):
            if isinstance(msg, ResultMessage):
                last_msg = msg

        if last_msg is None:
            raise ValueError("No message received from Claude Code.")

        result = {
            "session_id": last_msg.session_id or "",
            "last_message": asdict(last_msg),
            "result": last_msg.result,
        }
        output_schema = kwargs.get("output_schema") or self.kwargs.get("output_schema")
        if output_schema and last_msg.structured_output:
            result["structured_output"] = last_msg.structured_output
        return result

    async def reply_stream(self, inputs: Any, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        """Stream Claude Code events as unified StreamChunk objects."""
        from claude_agent_sdk import query, ResultMessage, AssistantMessage, StreamEvent, UserMessage
        from claude_agent_sdk.types import RateLimitEvent

        opts = self._build_options(inputs, stream=True, **kwargs)

        block_ids: dict[int, str] = {}
        block_types: dict[int, str] = {}
        tool_call_names: dict[int, str] = {}
        visible_tool_call_ids: set[str] = set()
        current_session_id: str | None = None
        emitted_content = False
        received_result_message = False

        stream = query(prompt=inputs, options=opts)
        try:
            async for msg in stream:
                if isinstance(msg, StreamEvent):
                    current_session_id = msg.session_id or current_session_id
                    chunk = self._raw_event_to_chunk(
                        msg.event,
                        session_id=msg.session_id,
                        block_ids=block_ids,
                        block_types=block_types,
                        tool_call_names=tool_call_names,
                    )
                    if chunk is not None:
                        chunk.session_id = chunk.session_id or msg.session_id
                        if chunk.chunk_type == ChunkEnum.TOOL_CALL and chunk.tool_call_id:
                            visible_tool_call_ids.add(chunk.tool_call_id)
                        if chunk.chunk_type == ChunkEnum.CONTENT and chunk.chunk:
                            emitted_content = True
                        yield chunk

                elif isinstance(msg, UserMessage):
                    for chunk in self._message_content_to_chunks(msg, current_session_id, visible_tool_call_ids):
                        yield chunk

                elif isinstance(msg, ResultMessage):
                    received_result_message = True
                    current_session_id = msg.session_id or current_session_id
                    if not emitted_content and getattr(msg, "result", None):
                        emitted_content = True
                        yield self._chunk(ChunkEnum.CONTENT, session_id=msg.session_id or "", chunk=msg.result)
                    # Final result: emit USAGE + REPLY_END
                    meta = {
                        "duration_ms": msg.duration_ms,
                        "duration_api_ms": msg.duration_api_ms,
                        "stop_reason": msg.stop_reason,
                        "num_turns": msg.num_turns,
                    }
                    yield self._chunk(
                        ChunkEnum.USAGE,
                        session_id=msg.session_id or "",
                        chunk=json.dumps(msg.usage or {}),
                        metadata=meta,
                    )
                    if self._result_message_is_error(msg):
                        yield self._chunk(
                            ChunkEnum.ERROR,
                            session_id=msg.session_id or "",
                            chunk=str(msg.errors) if msg.errors else "Unknown error",
                        )
                    yield self._chunk(ChunkEnum.REPLY_END, session_id=msg.session_id or "", chunk="")

                elif isinstance(msg, AssistantMessage):
                    current_session_id = msg.session_id or current_session_id
                    # Intermediate assistant text/tool-use is already streamed
                    # via StreamEvents.  Still surface tool-result blocks if the
                    # SDK includes any in a regular assistant message.
                    for chunk in self._message_content_to_chunks(
                        msg,
                        current_session_id,
                        visible_tool_call_ids,
                        include_text=not emitted_content,
                    ):
                        if chunk.chunk_type == ChunkEnum.CONTENT and chunk.chunk:
                            emitted_content = True
                        yield chunk

                elif isinstance(msg, RateLimitEvent):
                    yield self._chunk(ChunkEnum.ERROR, session_id=msg.session_id, chunk="Rate limit exceeded")
        except Exception as exc:
            if received_result_message and self._is_trailing_success_error(exc):
                self.logger.debug(f"Ignoring Claude Code trailing success error after final result: {exc}")
            else:
                raise
        finally:
            try:
                await stream.aclose()
            except Exception as exc:
                if not (received_result_message and self._is_trailing_success_error(exc)):
                    raise
                self.logger.debug(f"Ignoring Claude Code stream close error after final result: {exc}")
