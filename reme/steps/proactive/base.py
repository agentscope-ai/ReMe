"""Shared workflow dependencies, authorization and read-only task queries."""

from ...components.proactive.manager import BaseAgentManager
from ...components.proactive.runtime import BaseAgentRuntime
from ...enumeration import ComponentEnum, ChunkEnum
from ...schema import AgentActivity, StreamChunk, collect_reply
from ..base_step import BaseStep, Ref
from ..file_io import check_path_permission


class BaseAgentStep(BaseStep):
    """Keep invocation helpers separate from Runtime state and service routing."""

    agent_manager = Ref(BaseAgentManager, ComponentEnum.AGENT_MANAGER)
    agent_runtime = Ref(BaseAgentRuntime, ComponentEnum.AGENT_RUNTIME)

    async def query_reply(self, agent_name, query, *, on_chunk=None):
        """Answer from a temporary read-only fork of the latest task session."""
        runtime = await self.agent_manager.resolve(agent_name)
        options = {"job_tools": [], "skills": [], "sandbox": "read-only", "approval_mode": "deny_all"}
        async with runtime.query_session(options) as session_id:

            async def observe(chunk):
                if chunk.chunk_type == ChunkEnum.ERROR and chunk.metadata.get("will_retry") is not True:
                    self.context.stream_error = str(chunk.chunk)
                if on_chunk is not None:
                    await on_chunk(chunk.model_copy(update={"session_id": chunk.session_id or session_id}))

            return await collect_reply(
                runtime.agent_wrapper.reply_stream(query, resume=session_id, **options),
                on_chunk=observe,
            )

    def require_paths(self, *paths):
        """Check invocation permissions before accessing workspace-owned resources."""
        if any(
            not check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")) for path in paths
        ):
            raise PermissionError("no permission to access agent runtime files")

    def parameters(self):
        """Exclude trusted framework context from public tool request validation."""
        injected = {"metadata", "_allowed_paths", "_agent_name", "_inbox_turn", "_automatic_notebook", "_on_chunk"}
        return {key: value for key, value in self.context.data.items() if key not in injected}


class ActivityTracker:
    """Reduce portable events while retaining every unfinished tool invocation."""

    def __init__(self, activity: AgentActivity) -> None:
        self.activity = activity
        self._tools: dict[str, str | None] = {}
        self._finished: set[str] = set()
        self._done = False

    def update(self, chunk: StreamChunk) -> AgentActivity:
        """Ignore duplicate results and transport DONE; only turn outcomes finish execution."""
        if self._done:
            return self.activity
        phase, tool = self.activity.phase, None
        kind, call_id = chunk.chunk_type, chunk.tool_call_id
        if kind in {"reply_start", "think"}:
            phase = "thinking"
            if self._tools:
                phase = "working"
                tool = next(reversed(self._tools.values()))
        elif kind == "content" and isinstance(chunk.chunk, str) and chunk.chunk:
            phase = "answering"
        elif kind == "tool_call" or (kind == "tool_result" and chunk.status == "running"):
            if call_id in self._finished:
                return self.activity
            tool = chunk.tool_call_name or self._tools.get(call_id)
            if call_id is not None:
                self._tools[call_id] = tool
            phase = "working"
        elif kind == "tool_result":
            if call_id is not None:
                if call_id not in self._tools:
                    return self.activity
                self._tools.pop(call_id)
                self._finished.add(call_id)
            tool = next(reversed(self._tools.values())) if self._tools else None
            phase = "working" if self._tools else "thinking"
        elif kind == "approval" and chunk.status == "waiting":
            phase = "waiting"
        elif kind == "reply_end":
            phase = chunk.status if chunk.status in {"completed", "failed", "interrupted", "waiting"} else phase
        elif kind == "error" and chunk.metadata.get("will_retry") is not True:
            phase = "failed"
        else:
            return self.activity
        if phase in {"thinking", "working"} and tool is not None:
            phase = (
                "remembering"
                if any(word in tool.lower() for word in ("memory", "conversation", "recall", "inbox", "reminders"))
                else "working"
            )
        if phase in {"completed", "failed", "interrupted"}:
            self._tools.clear()
            self._finished.clear()
            self._done = kind == "reply_end"
        if phase == self.activity.phase and tool == self.activity.tool:
            return self.activity
        self.activity = self.activity.model_copy(update={"phase": phase, "tool": tool})
        return self.activity
