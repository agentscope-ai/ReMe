"""One fixed Chat Agent with durable native session history."""

from .base import BaseAgentStep
from ....components import R


class BaseChatAgentStep(BaseAgentStep):
    """Share prompt, memory and tool bindings across inference and empty-session creation."""

    def session_options(self):
        """Build the shared Chat prompt, memory path and tool bindings."""
        runtime = self.agent_runtime
        memory = runtime.agents.path("chat") / "MEMORY.md"
        return {
            **self.web_options(True),
            **self.memory_options(),
            "system_prompt": self.prompt_format(
                "system_prompt",
                workspace_dir=self.workspace_path,
                memory_path=memory,
            ),
            "job_tools": self.kwargs.get("job_tools", []),
            "injected_job_kwargs": {"_agent_id": "chat"},
        }

    async def ensure_session(self, options, *, new=False):
        """Select the current Chat session or create a fresh one with the same bindings."""
        runtime = self.agent_runtime
        return await runtime.sessions.ensure("chat", runtime.agent_wrapper, options, runtime.now(), new=new)


@R.register("chat_agent_step")
class ChatAgentStep(BaseChatAgentStep):
    """Append a user query to chat's current session; the caller supplies no native ID."""

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"query"} or not isinstance(arguments["query"], str) or not arguments["query"].strip():
            raise ValueError("chat accepts only a non-empty query")
        self.require_paths(self.workspace_path)
        async with self.agent_runtime.chat_turn():
            options = self.session_options()
            session_id = await self.ensure_session(options)
            await self.stream_reply(arguments["query"], session_id=session_id, agent_id="chat", **options)
        return self.context.response


@R.register("new_chat_session_step")
class NewChatSessionStep(BaseChatAgentStep):
    """Select a fresh chat session and retain every previous native ID."""

    async def execute(self):
        if self.parameters():
            raise ValueError("new_chat_session accepts no arguments")
        self.require_paths(self.workspace_path)
        if self.agent_runtime.chat_lock.locked():
            raise RuntimeError("Chat Agent is busy; retry new_chat_session after its turn completes")
        async with self.agent_runtime.chat_turn():
            await self.ensure_session(self.session_options(), new=True)
        self.context.response.answer = {"agent_id": "chat", "created": True}
        return self.context.response


@R.register("list_chat_sessions_step")
class ListChatSessionsStep(BaseAgentStep):
    """Global history query; cold listing never creates a session."""

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) - {"offset", "limit"}:
            raise ValueError("list_chat_sessions accepts only offset and limit")
        self.require_paths(self.agent_runtime.agents.path("chat"))
        self.context.response.answer = await self.agent_runtime.list_chat_sessions(**arguments)
        return self.context.response
