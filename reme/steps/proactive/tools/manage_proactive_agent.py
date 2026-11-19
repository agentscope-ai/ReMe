"""Named task management with action-specific results for model callers."""

from ..agent import BaseAgentStep
from ....components import R
from ....schema.proactive_agent import ManageProactiveAgentRequest


@R.register("manage_proactive_agent_step")
class ManageProactiveAgentStep(BaseAgentStep):
    """Validate management arguments before accessing configuration or sessions."""

    async def execute(self):
        runtime = self.agent_runtime
        request = ManageProactiveAgentRequest.model_validate(self.parameters())
        self.require_paths(runtime.agents.directory)
        if request.action in {"create", "update", "query"}:
            self.require_paths(self.workspace_path)
        settings = request.model_dump(
            exclude_unset=True,
            exclude={"action", "agent_name", "task_description", "question"},
        )
        if request.action == "list":
            listing = await runtime.list_views()
            items = [item for item in listing["items"] if item["agent_name"] != "Remy"]
            answer = {"items": items, "total": len(items)}
        elif request.action in {"create", "status", "update"}:
            if request.action == "create":
                await runtime.create(
                    {"agent_name": request.agent_name, "task_description": request.task_description, **settings},
                )
            elif request.action == "update":
                await runtime.update(request.agent_name, settings)
            answer = await runtime.view_agent(request.agent_name)
        elif request.action == "delete":
            await runtime.delete(request.agent_name)
            answer = f"Deleted task '{request.agent_name}'; history files retained."
        else:
            response = await self.run_job(
                self.kwargs.get("query_job", "query_proactive_agent"),
                agent_name=request.agent_name,
                query=request.question,
                _allowed_paths=self.context.get("_allowed_paths"),
            )
            self.context.response.success = response.success
            answer = response.answer["answer"] if response.success else response.answer
        self.context.response.answer = answer
        return self.context.response
