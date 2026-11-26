"""Tools available only to the active reminder fork."""

from ....constants import CHAT_AGENT_NAME
from ..agent.base import BaseAgentStep
from ....core.registry import R
from ....schema.proactive_runtime import SubmitChatReminderRequest


@R.register("submit_chat_reminder_step")
class SubmitChatReminderStep(BaseAgentStep):
    """Send an assistant reminder or commit an explicit empty decision."""

    async def execute(self):
        request = SubmitChatReminderRequest.model_validate(self.parameters())
        self.require_paths(self.workspace_path)
        if self.context.get("_agent_name") != CHAT_AGENT_NAME:
            raise ValueError("inbox tools require the calling Chat Agent")
        notifications = self.proactive_manager.notifications
        turn = notifications.current_turn(self.context.get("_inbox_turn"))
        self.context.response.answer = await notifications.send(turn, request.content)
        return self.context.response
