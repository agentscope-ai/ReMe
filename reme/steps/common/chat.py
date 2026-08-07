"""Workspace-aware streaming chat step for the ReMe web interface."""

import datetime
import zoneinfo

from ..base_step import BaseStep
from ...components import R
from ...enumeration import ChunkEnum


@R.register("chat_step")
class ChatStep(BaseStep):
    """Stream a read-only agent conversation over the current workspace."""

    DEFAULT_SYSTEM_PROMPT = """You are ReMe Agent, an assistant for a local-first memory workspace.
Use the available ReMe tools when workspace facts are needed. Cite workspace-relative file paths
when referring to notes. Never invent file contents, and do not claim to have changed files because
this chat intentionally provides read-only tools. Reply in the user's language."""
    READ_ONLY_TOOLS = ["search", "list", "stat", "traverse"]

    def _system_prompt(self) -> str:
        """Append request-time environment facts to the configured prompt."""
        timezone = self.app_context.app_config.timezone if self.app_context is not None else None
        try:
            current = datetime.datetime.now(zoneinfo.ZoneInfo(timezone)) if timezone else datetime.datetime.now()
        except (zoneinfo.ZoneInfoNotFoundError, ValueError):
            current = datetime.datetime.now()

        assert self.context is not None
        base_prompt = str(self.context.get("system_prompt") or self.DEFAULT_SYSTEM_PROMPT).rstrip()
        return (
            f"{base_prompt}\n\n"
            "<environment_context>\n"
            f"Current date: {current.date().isoformat()}\n"
            f"Current working directory: {self.agent_wrapper.cwd.resolve(strict=False)}\n"
            "</environment_context>"
        )

    async def execute(self):
        assert self.context is not None
        query = str(self.context.get("query") or "").strip()
        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response
        if self.agent_wrapper is None:
            raise RuntimeError("chat_step requires an agent_wrapper")

        wrapper_kwargs = {
            "system_prompt": self._system_prompt(),
            "job_tools": self.READ_ONLY_TOOLS,
            "builtin_tools": [],
        }
        if session_id := self.context.get("session_id"):
            wrapper_kwargs["resume"] = str(session_id)

        parts: list[str] = []
        async for chunk in self.agent_wrapper.reply_stream(query, **wrapper_kwargs):
            if chunk.chunk_type == ChunkEnum.REPLY_END:
                answer = "".join(parts).strip()
                if answer:
                    chunk.metadata["answer"] = answer
            await self.context.add_stream_chunk(chunk)
            if chunk.chunk_type == ChunkEnum.CONTENT and isinstance(chunk.chunk, str):
                parts.append(chunk.chunk)
            if chunk.session_id:
                self.context.response.metadata["session_id"] = chunk.session_id

        self.context.response.success = True
        self.context.response.answer = "".join(parts).strip()
        return self.context.response
