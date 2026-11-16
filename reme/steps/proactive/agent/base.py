"""Typed runtime access and workspace authorization shared by agent steps."""

import asyncio
from uuid import uuid4

from ...base_step import BaseStep, Ref
from ...file_io import check_path_permission
from ....components.agent_runtime import BaseAgentRuntime
from ....enumeration import ComponentEnum
from ....schema import AgentActivity, Response, collect_reply
from ....utils.agent_activity import ActivityTracker


class BaseAgentStep(BaseStep):
    """Keep request handlers stateless; the application component owns the runtime."""

    agent_runtime: BaseAgentRuntime = Ref(BaseAgentRuntime, ComponentEnum.AGENT_RUNTIME)

    async def stream_reply(self, query: str, *, session_id: str, agent_id: str, **options) -> Response:
        """Relay a formal conversation and preserve its terminal outcome and native metadata."""
        runtime = self.agent_runtime
        tracker = ActivityTracker(
            AgentActivity(
                session_id=session_id,
                execution_session_id=session_id,
                turn_id=uuid4().hex,
                phase="thinking",
                started_at=runtime.now(),
            ),
        )
        runtime.activities[agent_id] = tracker.activity

        async def observe(chunk):
            if chunk.turn_id:
                tracker.activity = tracker.activity.model_copy(
                    update={
                        "turn_id": chunk.turn_id,
                        "execution_session_id": chunk.session_id or session_id,
                    },
                )
            runtime.activities[agent_id] = tracker.update(chunk)
            publish(chunk)
            await self.context.add_stream_chunk(chunk)

        try:
            with runtime.live_turns.observe(agent_id, session_id, query) as publish:
                result = await collect_reply(
                    runtime.agent_wrapper.reply_stream(query, resume=session_id, **options),
                    on_chunk=observe,
                )
        except BaseException as exc:
            phase = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
            runtime.activities[agent_id] = tracker.activity.model_copy(update={"phase": phase, "tool": None})
            raise
        response = self.context.response
        response.success = result["status"] == "completed" and self.context.stream_error is None
        response.answer = result["result"]
        response.metadata.update(result, agent_id=agent_id)
        return response

    @staticmethod
    def web_options(online: bool):
        """Apply one network policy to search, URL extraction and commands."""
        return {
            "job_tools": ["web_extract"] if online else [],
            "config": {
                "web_search": "live" if online else "disabled",
                "sandbox_workspace_write": {"network_access": online},
            },
        }

    def require_paths(self, *paths):
        """Preserve fail-closed tool bindings before accessing runtime files."""
        if any(
            not check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")) for path in paths
        ):
            raise PermissionError("no permission to access agent runtime files")

    def memory_options(self, agent_id="chat"):
        """Protect the caller's memory without restricting collectors or reminders."""
        registry = self.agent_runtime.agents
        return {"read_only_paths": [str(registry.path(agent_id) / "MEMORY.md")]}

    def parameters(self):
        """Separate server-injected context from public request arguments."""
        return {
            key: value
            for key, value in self.context.data.items()
            if key not in {"metadata", "_allowed_paths", "_agent_id", "_inbox_turn"}
        }
