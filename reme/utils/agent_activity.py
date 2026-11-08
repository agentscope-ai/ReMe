"""Turn-scoped execution facts for presentation, independent of transcript text."""

from ..schema.agent_activity import AgentActivity
from ..schema.stream_chunk import StreamChunk


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
                if any(word in tool.lower() for word in ("memory", "conversation", "recall", "read_proactive_inbox"))
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
