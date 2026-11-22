"""Current Agent conversations independent of native wrapper or desktop implementation."""

from contextlib import aclosing

from ....constants import CHAT_AGENT_NAME
from .base import BaseAgentStep
from .chat import BaseChatAgentStep
from ....utils.files.path_lock import get_path_lock
from ....core.registry import R
from ....enumeration import ChunkEnum
from ....schema import StreamChunk
from ....utils.agent_transcript import TranscriptProjector
from ....schema.agent_view import (
    AgentChatRequest,
    AgentRequest,
    AgentInboxRequest,
    AgentOutboxRequest,
    AgentReminderRequest,
    AgentTaskRequest,
)


class _AgentViewStep(BaseAgentStep):
    """Validate identities and permissions before reading runtime resources."""

    def request(self, schema=AgentRequest):
        """Validate public inputs before authorizing the workspace."""
        request = schema.model_validate(self.parameters())
        self.require_paths(self.workspace_path)
        return request


@R.register("agent_list_step")
class AgentListStep(_AgentViewStep):
    """List Remy and retained task identities without creating sessions."""

    async def execute(self):
        if self.parameters():
            raise ValueError("list_agents accepts no arguments")
        self.require_paths(self.workspace_path)
        self.context.response.answer = await self.agent_runtime.list_views()
        return self.context.response


@R.register("agent_session_step")
class AgentSessionStep(_AgentViewStep):
    """Read Chat history or the latest task turn without changing the formal session."""

    async def execute(self):
        request = self.request()
        self.context.response.answer = await self.agent_runtime.read_agent(request.agent_name)
        return self.context.response


@R.register("agent_task_step")
class AgentTaskStep(_AgentViewStep):
    """Reuse contained file operations for a named task's original description."""

    async def execute(self):
        request = AgentTaskRequest.model_validate(self.parameters())
        runtime = self.agent_runtime
        self.require_paths(runtime.agents.directory)
        agent = await runtime.definition(request.agent_name)
        target = runtime.agents.task_path(agent)
        self.require_paths(target)
        path = target.relative_to(self.workspace_path).as_posix()
        if request.action == "read":
            async with await get_path_lock(target):
                await self.dispatch_steps(["load_step"], path=path)
        else:
            await self.dispatch_steps(
                ["save_step"],
                path=path,
                content=request.content,
                expected_mtime=request.expected_mtime,
            )
        response = self.context.response
        if response.success:
            response.answer = {
                "path": path,
                "mtime": response.metadata["mtime"],
                **({"content": response.answer} if request.action == "read" else {}),
            }
        return response


@R.register("agent_events_step")
class AgentEventsStep(_AgentViewStep):
    """Relay Chat history changes or atomic latest-task-turn snapshots with sidebar status."""

    async def execute(self):
        request = self.request()
        async with aclosing(self.agent_runtime.watch_agent(request.agent_name)) as events:
            async for event in events:
                await self.context.add_stream_chunk(StreamChunk(chunk_type=ChunkEnum.DATA, chunk=event))
        return self.context.response


@R.register("agent_chat_step")
class AgentChatStep(_AgentViewStep, BaseChatAgentStep):
    """Continue Chat or ask a read-only fork of the latest task session."""

    async def execute(self):
        request = self.request(AgentChatRequest)
        if not request.query.strip():
            raise ValueError("query must not be blank")
        projector = TranscriptProjector()

        async def publish(chunk):
            projector.update(chunk)
            if projector.error:
                self.context.stream_error = projector.error
            # Main Chat content has one authority: the session subscription.
            if request.agent_name == CHAT_AGENT_NAME and chunk.chunk_type not in {
                ChunkEnum.REPLY_START,
                ChunkEnum.REPLY_END,
                ChunkEnum.ERROR,
            }:
                return
            await self.context.add_stream_chunk(
                StreamChunk(chunk_type=ChunkEnum.DATA, chunk=projector.message().model_dump(mode="json")),
            )

        if request.agent_name == CHAT_AGENT_NAME:
            return await self.chat(request.query, on_chunk=publish)
        reply = await self.query_reply(request.agent_name, request.query, on_chunk=publish)
        self.context.response.success = reply["status"] == "completed" and self.context.stream_error is None
        self.context.response.answer = reply["result"]
        self.context.response.metadata.update(reply, agent_name=request.agent_name)
        return self.context.response


@R.register("agent_new_session_step")
class AgentNewSessionStep(_AgentViewStep, BaseChatAgentStep):
    """Start a fresh main Chat session while retaining earlier history."""

    async def execute(self):
        if self.parameters():
            raise ValueError("create_chat_session accepts no arguments")
        self.require_paths(self.workspace_path)
        await self.new_session()
        self.context.response.answer = await self.agent_runtime.read_agent(CHAT_AGENT_NAME)
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
    """Read a task reminder or the sources of a Chat message."""

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
