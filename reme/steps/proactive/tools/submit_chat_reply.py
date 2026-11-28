"""Submit one decision using the active fork's server-owned capability."""

from ..base import BaseAgentStep
from ....constants import CHAT_AGENT_NAME
from ....core.registry import R
from ....schema.proactive_runtime import SubmitChatReplyRequest


@R.register("submit_chat_reply_step")
class SubmitChatReplyStep(BaseAgentStep):
    """A foreground call or an expired fork cannot acknowledge notifications."""

    async def execute(self):
        request = SubmitChatReplyRequest.model_validate(self.parameters())
        self.require_paths(self.workspace_path)
        if self.context.get("_agent_name") != CHAT_AGENT_NAME:
            raise ValueError("Chat reply requires the calling Remy notification fork")
        self.context.response.answer = await self.agent_runtime.submit_reply(
            self.context.get("_inbox_turn"),
            request.action,
            request.content,
        )
        return self.context.response
