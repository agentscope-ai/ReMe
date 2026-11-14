"""One fixed main Agent with durable native session history."""

from .base import BaseAgentStep
from ....components import R
from ....schema import collect_reply


class _ChatAgentStep(BaseAgentStep):
    """Share prompt, memory and tool bindings across inference and empty-session creation."""

    def _options(self):
        runtime = self.agent_runtime
        memory = runtime.agents.path("main") / "MEMORY.md"
        options = self.web_options(True)
        return {
            **options,
            "system_prompt": self.prompt_format(
                "system_prompt",
                workspace_dir=self.workspace_path,
                memory_path=memory,
                reminder_policy=self.get_prompt("reminder_policy"),
            ),
            "job_tools": self.kwargs.get("job_tools", []),
            "injected_job_kwargs": {"_agent_id": "main"},
        }

    async def _session(self, options, *, new=False):
        runtime = self.agent_runtime
        return await runtime.sessions.ensure("main", runtime.agent_wrapper, options, runtime.now(), new=new)


@R.register("chat_agent_step")
class ChatAgentStep(_ChatAgentStep):
    """Append a user query to main's current session; the caller supplies no native ID."""

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"query"} or not isinstance(arguments["query"], str) or not arguments["query"].strip():
            raise ValueError("chat accepts only a non-empty query")
        self.require_paths(self.workspace_path)
        async with self.agent_runtime.main_turn():
            options = self._options()
            session_id = await self._session(options)
            result = await collect_reply(
                self.agent_runtime.agent_wrapper.reply_stream(
                    arguments["query"],
                    resume=session_id,
                    **options,
                ),
                on_chunk=self.context.add_stream_chunk,
            )
            self.context.response.success = result["status"] == "completed" and self.context.stream_error is None
            self.context.response.answer = result["result"]
            self.context.response.metadata.update(result)
            self.context.response.metadata["agent_id"] = "main"
        return self.context.response


@R.register("inbox_chat_step")
class InboxChatStep(_ChatAgentStep):
    """Wake the same Chat session to read messages and decide whether to notify."""

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) != {"priority", "msg_ids"} or arguments["priority"] not in {"P0", "P1", "P2"}:
            raise ValueError("inbox_chat requires priority and msg_ids")
        self.require_paths(self.workspace_path)
        runtime = self.agent_runtime
        async with runtime.main_turn():
            priority = arguments["priority"]
            idle = max(0, runtime.now().timestamp() - runtime.inbox.last_activity_at.timestamp())
            if priority != "P0" and idle < getattr(runtime.inbox.config, f"{priority.lower()}_idle_seconds"):
                self.context.response.answer = {"skipped": "not_idle"}
                return self.context.response
            if not any(identifier in runtime.inbox.pending for identifier in arguments["msg_ids"]):
                self.context.response.answer = {"skipped": "already_read"}
                return self.context.response
            options = self._options()
            session_id = await self._session(options)
            status = await runtime.agent_wrapper.get_session_status(session_id)
            if status.status not in {"idle", "not_loaded"}:
                self.context.response.answer = {"skipped": "busy"}
                return self.context.response
            reply = await runtime.agent_wrapper.reply(
                self.prompt_format("inbox_hint", priority=priority, urgent=priority == "P0", normal=priority != "P0"),
                resume=session_id,
                activity_source="task",
                activity_task_id="inbox",
                **options,
            )
            self.context.response.success = reply["status"] == "completed"
            self.context.response.answer = reply["result"] if self.context.response.success else reply.get("error")
        return self.context.response


@R.register("new_chat_session_step")
class NewChatSessionStep(_ChatAgentStep):
    """Select a fresh main session and retain every previous native ID."""

    async def execute(self):
        if self.parameters():
            raise ValueError("new_chat_session accepts no arguments")
        self.require_paths(self.workspace_path)
        if self.agent_runtime.chat_lock.locked():
            raise RuntimeError("main Agent is busy; retry new_chat_session after its turn completes")
        async with self.agent_runtime.main_turn():
            await self._session(self._options(), new=True)
        self.context.response.answer = {"agent_id": "main", "created": True}
        return self.context.response


@R.register("list_chat_sessions_step")
class ListChatSessionsStep(BaseAgentStep):
    """Global history query; cold listing never creates a session."""

    async def execute(self):
        arguments = self.parameters()
        if set(arguments) - {"offset", "limit"}:
            raise ValueError("list_chat_sessions accepts only offset and limit")
        self.require_paths(self.agent_runtime.agents.path("main"))
        self.context.response.answer = await self.agent_runtime.list_chat_sessions(**arguments)
        return self.context.response
