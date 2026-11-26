"""Named task management with action-specific results for model callers."""

from ....constants import CHAT_AGENT_NAME
from ..agent import BaseAgentStep
from ....core.registry import R
from ....schema.proactive_agent import ManageProactiveAgentRequest


@R.register("manage_proactive_agent_step")
class ManageProactiveAgentStep(BaseAgentStep):
    """Validate management arguments before accessing configuration or sessions."""

    async def execute(self):
        manager = self.proactive_manager
        request = ManageProactiveAgentRequest.model_validate(self.parameters())
        self.require_paths(manager.registry.directory)
        if request.action in {"create", "update", "query"}:
            self.require_paths(self.workspace_path)
        settings = request.model_dump(
            exclude_unset=True,
            exclude={"action", "agent_name", "task_description", "question"},
        )
        if request.action == "list":
            listing = await manager.list_views()
            items = [item for item in listing["items"] if item["agent_name"] != CHAT_AGENT_NAME]
            answer = {"items": items, "total": len(items)}
        elif request.action in {"create", "status", "update"}:
            status = None
            if request.action == "create":
                status = await manager.create(
                    {"agent_name": request.agent_name, "task_description": request.task_description, **settings},
                )
            elif request.action == "update":
                status = await manager.update(request.agent_name, settings)
            answer = await manager.view_agent(request.agent_name, status=status)
        elif request.action == "delete":
            await manager.delete(request.agent_name)
            answer = f"Deleted task '{request.agent_name}'; history files retained."
        else:
            reply = await self.query_reply(request.agent_name, request.question)
            self.context.response.success = reply["status"] == "completed"
            answer = (
                reply["result"]
                if self.context.response.success
                else reply.get("error") or f"Proactive query {reply['status']}"
            )
        self.context.response.answer = answer
        return self.context.response
