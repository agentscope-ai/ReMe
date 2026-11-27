"""Remy's foreground conversation workflow and native output relay."""

import asyncio
from uuid import uuid4

from ..base import BaseAgentStep, ActivityTracker
from ....core.registry import R
from ....enumeration import ChunkEnum
from ....schema import AgentActivity, collect_reply
from ....schema.agent_view import AskAgentRequest


class BaseChatAgentStep(BaseAgentStep):
    """Prepare the formal Chat binding independently of a reminder fork's tools."""

    skill_names = ("manage-proactive-runtime", "manage-connector", "manage-notebook")
    job_tools = ("manage_proactive_runtime", "manage_connector", "manage_notebook")

    def session_options(self):
        """Build the formal role binding independently of temporary fork tools."""
        runtime = self.agent_runtime
        config = self.app_context.app_config.jobs.get(runtime.run_job)
        if config is None or not config.steps:
            raise ValueError("Remy's execution Job must be configured")
        binding = config.steps[0].model_dump()
        options = {
            "job_tools": list(binding.get("job_tools", self.job_tools) or []),
            "skills": self.get_skills(runtime.name, binding.get("skills", self.skill_names)),
            "system_prompt": self.prompt_format(
                "system_prompt",
                workspace_dir=self.workspace_path,
                memory_path=runtime.memory_path,
            ),
            "injected_job_kwargs": {"_agent_name": runtime.name},
        }
        for key in ("effort", "sandbox"):
            if binding.get(key) is not None:
                options[key] = binding[key]
        return options

    async def ensure_session(self, *, new=False):
        """Prepare a primary session using the role's execution binding."""
        return await self.agent_runtime.ensure_session(self.session_options(), new=new)

    async def new_session(self):
        """Rotate the foreground conversation only while Chat is idle."""
        runtime = self.agent_runtime
        if runtime.busy:
            raise RuntimeError("Chat Agent is busy; retry after its turn completes")
        async with runtime.chat_turn():
            return await self.ensure_session(new=True)

    async def chat(self, query, *, on_chunk=None):
        """Run one foreground turn within a scoped set of native skills."""
        runtime = self.agent_runtime
        async with runtime.chat_turn():
            options = self.session_options()
            async with runtime.agent_wrapper.skill_scope(**options) as options:
                session_id = await runtime.ensure_session(options)
                return await self.stream_reply(query, session_id=session_id, on_chunk=on_chunk, **options)

    async def stream_reply(self, query, *, session_id, on_chunk=None, **options):
        """Relay portable model events while retaining the Runtime live snapshot."""
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
        runtime.activity = tracker.activity

        async def observe(chunk):
            if chunk.chunk_type == ChunkEnum.ERROR and chunk.metadata.get("will_retry") is not True:
                self.context.stream_error = str(chunk.chunk)
            if chunk.turn_id:
                tracker.activity = tracker.activity.model_copy(
                    update={"turn_id": chunk.turn_id, "execution_session_id": chunk.session_id or session_id},
                )
            runtime.activity = tracker.update(chunk)
            publish(chunk)
            value = chunk.model_copy(update={"session_id": chunk.session_id or session_id})
            if on_chunk is not None:
                await on_chunk(value)
            elif self.context.stream:
                await self.context.add_stream_chunk(value)

        try:
            with runtime.live_turns.observe(runtime.name, session_id, query) as publish:
                result = await collect_reply(
                    runtime.agent_wrapper.reply_stream(query, resume=session_id, **options),
                    on_chunk=observe,
                )
        except BaseException as exc:
            phase = "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed"
            runtime.activity = tracker.activity.model_copy(update={"phase": phase, "tool": None})
            raise
        response = self.context.response
        response.success = result["status"] == "completed" and self.context.stream_error is None
        response.answer = result["result"]
        response.metadata.update(result, agent_name=runtime.name)
        return response


@R.register("chat_agent_step")
class ChatAgentStep(BaseChatAgentStep):
    """Run one foreground turn; the service stream owns transport termination."""

    async def execute(self):
        request = AskAgentRequest.model_validate(self.parameters())
        if request.agent_name != self.agent_runtime.name or not request.query.strip():
            raise ValueError("Chat execution requires Remy and a nonblank query")
        self.require_paths(self.workspace_path)
        return await self.chat(request.query, on_chunk=self.context.get("_on_chunk"))
