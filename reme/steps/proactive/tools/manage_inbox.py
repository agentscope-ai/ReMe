"""Read and mark messages in the calling Agent's durable inbox."""

from ..agent import BaseAgentStep
from ....components import R
from ....schema.agent_runtime import ManageInboxRequest


@R.register("manage_inbox_step")
class ManageInboxStep(BaseAgentStep):
    """Resolve receiver identity before accessing any inbox files."""

    async def execute(self):
        request = ManageInboxRequest.model_validate(self.parameters())
        caller = self.context.get("_agent_id")
        agent_id = request.agent_id or caller
        if agent_id is None:
            raise ValueError("agent_id is required outside an Agent-bound tool call")
        if caller is not None and caller != agent_id:
            raise ValueError("inbox agent_id does not match the calling Agent")
        inbox = self.agent_runtime.inbox
        self.require_paths(inbox.directory(agent_id))
        self.require_paths(self.agent_runtime.agents.directory)
        if request.action == "read":
            answer = await inbox.read(agent_id, request.priority, request.unread_only, request.limit)
        else:
            answer = await inbox.mark_read(agent_id, request.msg_ids)
        self.context.response.answer = answer
        return self.context.response
