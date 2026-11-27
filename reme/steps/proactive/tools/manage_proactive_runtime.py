"""Manage background Runtime identities without owning their lifecycle."""

from ..base import BaseAgentStep
from ..agent.initialize import initialize_task
from ....core.registry import R
from ....schema.proactive_agent import ManageProactiveRuntimeRequest


@R.register("manage_proactive_runtime_step")
class ManageProactiveRuntimeStep(BaseAgentStep):
    """Validate the action, then delegate reliable operations to AgentManager."""

    async def execute(self):
        request = ManageProactiveRuntimeRequest.model_validate(self.parameters())
        manager = self.agent_manager
        self.require_paths(manager.registry.directory)
        if request.action in {"create", "update", "query"}:
            self.require_paths(self.workspace_path)
        values = request.model_dump(exclude_unset=True, exclude={"action", "agent_name", "question"})
        if request.action == "list":
            definitions = await manager.discover()
            items = []
            for name in definitions:
                try:
                    items.append(await manager.status(name))
                except (OSError, ValueError, RuntimeError) as exc:
                    items.append({"agent_name": name, "error": str(exc)})
            answer = {"items": items, "total": len(items)}
        elif request.action == "create":
            answer = await manager.create({"agent_name": request.agent_name, **values})
            runtime = await manager.resolve(request.agent_name)
            await initialize_task(runtime, self.app_context, request.task_description)
            answer = {**await manager.status(request.agent_name), "restored": answer["restored"]}
        elif request.action == "update":
            answer = await manager.update(request.agent_name, values)
        elif request.action == "delete":
            answer = await manager.delete(request.agent_name)
        elif request.action == "status":
            answer = await manager.status(request.agent_name)
        else:
            reply = await self.query_reply(request.agent_name, request.question)
            self.context.response.success = reply["status"] == "completed" and self.context.stream_error is None
            answer = (
                reply["result"]
                if self.context.response.success
                else self.context.stream_error or reply.get("error") or f"Task query {reply['status']}"
            )
        self.context.response.answer = answer
        return self.context.response
