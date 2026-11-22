"""Tools available only to the active reminder fork."""

from ....constants import CHAT_AGENT_NAME
from ..agent.base import BaseAgentStep
from ....core.registry import R
from ....schema.proactive_runtime import SubmitChatReminderRequest


class _ChatReminderTool(BaseAgentStep):
    """Bind calls to one runtime-owned batch, never a model-selected session."""

    def turn(self):
        """Validate scope and file permissions before accessing the active batch."""
        self.require_paths(self.workspace_path)
        if self.context.get("_agent_name") != CHAT_AGENT_NAME:
            raise ValueError("inbox tools require the calling Chat Agent")
        return self.proactive_runtime.notifications.current_turn(self.context.get("_inbox_turn"))


@R.register("read_chat_reminders_step")
class ReadChatRemindersStep(_ChatReminderTool):
    """Expose reminder text and source records while the runtime owns acknowledgements."""

    async def execute(self):
        if self.parameters():
            raise ValueError("read_chat_reminders accepts no arguments")
        batch = await self.proactive_runtime.notifications.read_turn(self.turn())
        self.context.response.answer = {
            "items": [
                {
                    "agent_name": item["from_agent_name"],
                    "delivered_at": item["delivered_at"],
                    "message": item["message"],
                    "paths": item["paths"].copy(),
                }
                for item in batch["items"]
            ],
        }
        return self.context.response


@R.register("submit_chat_reminder_step")
class SubmitChatReminderStep(_ChatReminderTool):
    """Send an assistant reminder or commit an explicit empty decision."""

    async def execute(self):
        request = SubmitChatReminderRequest.model_validate(self.parameters())
        self.context.response.answer = await self.proactive_runtime.notifications.send(self.turn(), request.content)
        return self.context.response
