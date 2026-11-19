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

    async def query_reply(self, agent_name, query, *, on_chunk=None):
        """Ask a disposable fork without changing task history or live execution state."""
        runtime = self.agent_runtime
        options = {"job_tools": [], "sandbox": "read-only", "approval_mode": "deny_all"}
        async with runtime.query_session(agent_name, options) as session_id:

            async def observe(chunk):
                await on_chunk(chunk.model_copy(update={"session_id": chunk.session_id or session_id}))

            return await collect_reply(
                runtime.agent_wrapper.reply_stream(
                    query,
                    resume=session_id,
                    **options,
                ),
                on_chunk=observe if on_chunk else self.context.add_stream_chunk if self.context.stream else None,
            )

    async def stream_reply(self, query: str, *, session_id: str, agent_name: str, on_chunk=None, **options) -> Response:
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
        runtime.activities[agent_name] = tracker.activity

        async def observe(chunk):
            if chunk.turn_id:
                tracker.activity = tracker.activity.model_copy(
                    update={
                        "turn_id": chunk.turn_id,
                        "execution_session_id": chunk.session_id or session_id,
                    },
                )
            runtime.activities[agent_name] = tracker.update(chunk)
            publish(chunk)
            if on_chunk:
                await on_chunk(chunk.model_copy(update={"session_id": chunk.session_id or session_id}))
            else:
                await self.context.add_stream_chunk(chunk)

        try:
            with runtime.live_turns.observe(agent_name, session_id, query) as publish:
                result = await collect_reply(
                    runtime.agent_wrapper.reply_stream(query, resume=session_id, **options),
                    on_chunk=observe,
                )
        except BaseException as exc:
            phase = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
            runtime.activities[agent_name] = tracker.activity.model_copy(update={"phase": phase, "tool": None})
            raise
        response = self.context.response
        response.success = result["status"] == "completed" and self.context.stream_error is None
        response.answer = result["result"]
        response.metadata.update(result, agent_name=agent_name)
        return response

    def require_paths(self, *paths):
        """Preserve fail-closed tool bindings before accessing runtime files."""
        if any(
            not check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")) for path in paths
        ):
            raise PermissionError("no permission to access agent runtime files")

    def parameters(self):
        """Separate server-injected context from public request arguments."""
        return {
            key: value
            for key, value in self.context.data.items()
            if key not in {"metadata", "_allowed_paths", "_agent_name", "_inbox_turn"}
        }
