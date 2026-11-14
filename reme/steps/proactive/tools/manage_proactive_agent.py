"""One structured facade for both kinds of proactive agents."""

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
        if request.action == "list":
            answer = await runtime.list_agents()
        elif request.action == "create":
            answer = await runtime.create(request.config)
        elif request.action == "status":
            answer = await runtime.status(request.agent_id)
        elif request.action == "update":
            answer = await runtime.update(request.agent_id, request.config)
        elif request.action == "delete":
            answer = await runtime.delete(request.agent_id)
        else:
            response = await self.run_job(
                self.kwargs.get("query_job", "proactive_query"),
                agent_id=request.agent_id,
                query=request.query,
                _allowed_paths=self.context.get("_allowed_paths"),
            )
            self.context.response.success = response.success
            answer = response.answer
        self.context.response.answer = answer
        return self.context.response
