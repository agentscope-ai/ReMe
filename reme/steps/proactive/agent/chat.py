"""One fixed Chat Agent with durable native session history."""

from ....constants import CHAT_AGENT_NAME
from .base import BaseAgentStep
from ....core.registry import R


class BaseChatAgentStep(BaseAgentStep):
    """Share prompt, memory and tool bindings across inference and empty-session creation."""

    def session_options(self):
        """Build the shared Chat prompt, memory path and tool bindings."""
        manager = self.proactive_manager
        memory = manager.registry.chat_memory_path()
        return {
            "job_tools": self.kwargs.get("job_tools") or [],
            "system_prompt": self.prompt_format(
                "system_prompt",
                workspace_dir=self.workspace_path,
                memory_path=memory,
            ),
            "injected_job_kwargs": {"_agent_name": CHAT_AGENT_NAME},
        }

    async def ensure_session(self, options, *, new=False):
        """Select the current Chat session or create a fresh one with the same bindings."""
        manager = self.proactive_manager
        return await manager.sessions.ensure(CHAT_AGENT_NAME, manager.agent_wrapper, options, manager.now(), new=new)

    async def chat(self, query, *, on_chunk):
        """Continue Remy's selected native session."""
        async with self.proactive_manager.chat_turn():
            options = self.session_options()
            session_id = await self.ensure_session(options)
            return await self.stream_reply(
                query,
                session_id=session_id,
                agent_name=CHAT_AGENT_NAME,
                on_chunk=on_chunk,
                **options,
            )

    async def new_session(self):
        """Rotate the selected Chat session under the same generation lock."""
        if self.proactive_manager.chat_lock.locked():
            raise RuntimeError("Chat Agent is busy; retry after its turn completes")
        async with self.proactive_manager.chat_turn():
            await self.ensure_session(self.session_options(), new=True)


@R.register("list_chat_sessions_step")
class ListChatSessionsStep(BaseAgentStep):
    """Global history query; cold listing never creates a session."""

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) - {"offset", "limit"}:
            raise ValueError("list_chat_sessions accepts only offset and limit")
        self.require_paths(self.proactive_manager.registry.path(CHAT_AGENT_NAME))
        self.context.response.answer = await self.proactive_manager.list_chat_sessions(**arguments)
        return self.context.response
