"""One fixed Chat Agent with durable native session history."""

from .base import BaseAgentStep
from ....components import R


class BaseChatAgentStep(BaseAgentStep):
    """Share prompt, memory and tool bindings across inference and empty-session creation."""

    def session_options(self):
        """Build the shared Chat prompt, memory path and tool bindings."""
        runtime = self.agent_runtime
        memory = runtime.agents.chat_memory_path()
        return {
            "job_tools": self.kwargs.get("job_tools") or [],
            "system_prompt": self.prompt_format(
                "system_prompt",
                workspace_dir=self.workspace_path,
                memory_path=memory,
            ),
            "injected_job_kwargs": {"_agent_name": "Remy"},
        }

    async def ensure_session(self, options, *, new=False):
        """Select the current Chat session or create a fresh one with the same bindings."""
        runtime = self.agent_runtime
        return await runtime.sessions.ensure("Remy", runtime.agent_wrapper, options, runtime.now(), new=new)

    async def chat(self, query, *, on_chunk=None):
        """Share execution between the internal job and the normalized client API."""
        async with self.agent_runtime.chat_turn():
            options = self.session_options()
            session_id = await self.ensure_session(options)
            return await self.stream_reply(
                query,
                session_id=session_id,
                agent_name="Remy",
                on_chunk=on_chunk,
                **options,
            )

    async def new_session(self):
        """Rotate the selected Chat session under the same generation lock."""
        if self.agent_runtime.chat_lock.locked():
            raise RuntimeError("Chat Agent is busy; retry after its turn completes")
        async with self.agent_runtime.chat_turn():
            await self.ensure_session(self.session_options(), new=True)


@R.register("chat_agent_step")
class ChatAgentStep(BaseChatAgentStep):
    """Append a user query to chat's current session; the caller supplies no native ID."""

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"query"} or not isinstance(arguments["query"], str) or not arguments["query"].strip():
            raise ValueError("chat accepts only a non-empty query")
        self.require_paths(self.workspace_path)
        return await self.chat(arguments["query"])


@R.register("new_chat_session_step")
class NewChatSessionStep(BaseChatAgentStep):
    """Select a fresh chat session and retain every previous native ID."""

    async def execute(self):
        if self.parameters():
            raise ValueError("new_chat_session accepts no arguments")
        self.require_paths(self.workspace_path)
        await self.new_session()
        self.context.response.answer = {"agent_name": "Remy", "created": True}
        return self.context.response


@R.register("list_chat_sessions_step")
class ListChatSessionsStep(BaseAgentStep):
    """Global history query; cold listing never creates a session."""

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) - {"offset", "limit"}:
            raise ValueError("list_chat_sessions accepts only offset and limit")
        self.require_paths(self.agent_runtime.agents.path("Remy"))
        self.context.response.answer = await self.agent_runtime.list_chat_sessions(**arguments)
        return self.context.response
