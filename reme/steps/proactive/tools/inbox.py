"""Tools available only to the active reminder fork."""

from ..agent.base import BaseAgentStep
from ....components import R
from ....schema.agent_runtime import ReadInboxRequest, SendChatMessageRequest


class _InboxTool(BaseAgentStep):
    """Bind calls to one runtime-owned batch, never a model-selected session."""

    def turn(self):
        """Validate scope and file permissions before accessing the active batch."""
        self.require_paths(self.workspace_path)
        if self.context.get("_agent_id") != "chat":
            raise ValueError("inbox tools require the calling Chat Agent")
        return self.agent_runtime.notifications.current_turn(self.context.get("_inbox_turn"))


@R.register("read_inbox_step")
class ReadInboxStep(_InboxTool):
    """Read the selected batch; the framework consumes successful decisions."""

    async def execute(self):
        request = ReadInboxRequest.model_validate(self.parameters())
        self.context.response.answer = await self.agent_runtime.notifications.read_turn(self.turn(), request.priority)
        return self.context.response


@R.register("send_chat_msg_step")
class SendChatMessageStep(_InboxTool):
    """Send an assistant reminder or commit an explicit empty decision."""

    async def execute(self):
        request = SendChatMessageRequest.model_validate(self.parameters())
        self.context.response.answer = await self.agent_runtime.notifications.send(self.turn(), request.content)
        return self.context.response
