"""Current Agent conversations independent of native wrapper or desktop implementation."""

from contextlib import aclosing

from .base import BaseAgentStep
from .chat import ChatAgentStep, NewChatSessionStep
from .proactive import ProactiveAgentStep
from ....components import R
from ....enumeration import ChunkEnum
from ....schema import StreamChunk
from ....schema.agent_view import (
    AgentChatRequest,
    AgentRequest,
    AgentInboxRequest,
    AgentOutboxRequest,
    AgentReminderRequest,
)


class _AgentViewStep(BaseAgentStep):
    """Validate identities and permissions before reading runtime resources."""

    def request(self, schema=AgentRequest):
        """Validate public inputs before authorizing the workspace."""
        request = schema.model_validate(self.parameters())
        self.require_paths(self.workspace_path)
        return request

    async def chat_call(self, step_class):
        """Reuse the chat job while preserving the caller's shared runtime inputs."""
        supplied = "agent_id" in self.context.data
        agent_id = self.context.data.pop("agent_id", None)
        try:
            return await step_class(
                app_context=self.app_context,
                **{**self.kwargs, "agent_runtime": self.agent_runtime},
            )(self.context)
        finally:
            if supplied:
                self.context.data["agent_id"] = agent_id


@R.register("agent_list_step")
class AgentListStep(_AgentViewStep):
    """List Remy and retained task identities without creating sessions."""

    async def execute(self):
        if self.parameters():
            raise ValueError("agent_list accepts no arguments")
        self.require_paths(self.workspace_path)
        self.context.response.answer = await self.agent_runtime.list_views()
        return self.context.response


@R.register("agent_session_step")
class AgentSessionStep(_AgentViewStep):
    """Read Chat history or the latest task turn without changing the formal session."""

    async def execute(self):
        request = self.request()
        self.context.response.answer = await self.agent_runtime.read_agent(request.agent_id)
        return self.context.response


@R.register("agent_events_step")
class AgentEventsStep(_AgentViewStep):
    """Relay Chat history changes or atomic latest-task-turn snapshots with sidebar status."""

    async def execute(self):
        request = self.request()
        async with aclosing(self.agent_runtime.watch_agent(request.agent_id)) as events:
            async for event in events:
                await self.context.add_stream_chunk(StreamChunk(chunk_type=ChunkEnum.DATA, chunk=event))
        return self.context.response


@R.register("agent_chat_step")
class AgentChatStep(_AgentViewStep):
    """Continue formal sessions without consuming event batches or creating query forks."""

    async def execute(self):
        request = self.request(AgentChatRequest)
        if not request.query.strip():
            raise ValueError("query must not be blank")
        if request.agent_id == "chat":
            return await self.chat_call(ChatAgentStep)
        runtime = self.agent_runtime
        async with runtime.agent_turn(request.agent_id):
            agent = await runtime.definition(request.agent_id)
            step = ProactiveAgentStep(app_context=self.app_context, agent_runtime=runtime)
            options = step.session_options(agent)
            session_id = await runtime.sessions.ensure(agent.agent_id, runtime.agent_wrapper, options, runtime.now())
            await self.stream_reply(request.query, session_id=session_id, agent_id=agent.agent_id, **options)
        return self.context.response


@R.register("agent_new_session_step")
class AgentNewSessionStep(_AgentViewStep):
    """Select an empty formal session while preserving all earlier references."""

    async def execute(self):
        request = self.request()
        runtime = self.agent_runtime
        if request.agent_id == "chat":
            await self.chat_call(NewChatSessionStep)
        else:
            async with runtime.agent_turn(request.agent_id):
                agent = await runtime.definition(request.agent_id)
                options = ProactiveAgentStep(app_context=self.app_context, agent_runtime=runtime).session_options(agent)
                await runtime.sessions.ensure(
                    agent.agent_id,
                    runtime.agent_wrapper,
                    options,
                    runtime.now(),
                    new=True,
                )
        self.context.response.answer = await runtime.read_agent(request.agent_id)
        return self.context.response


@R.register("agent_outbox_step")
class AgentOutboxStep(_AgentViewStep):
    """Page a task Agent's committed reminders."""

    async def execute(self):
        request = self.request(AgentOutboxRequest)
        self.context.response.answer = await self.agent_runtime.outbox.list(**request.model_dump())
        return self.context.response


@R.register("agent_reminder_step")
class AgentReminderStep(_AgentViewStep):
    """Keep user reading separate from Inbox processing."""

    async def execute(self):
        request = self.request(AgentReminderRequest)
        self.context.response.answer = await self.agent_runtime.outbox.read(**request.model_dump())
        return self.context.response


@R.register("agent_inbox_step")
class AgentInboxStep(_AgentViewStep):
    """Read any Agent's inputs without claiming or acknowledging them."""

    async def execute(self):
        request = self.request(AgentInboxRequest)
        self.context.response.answer = await self.agent_runtime.list_inputs(**request.model_dump())
        return self.context.response
