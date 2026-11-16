"""Named task management with action-specific results for model callers."""

from ..agent import BaseAgentStep
from ....components import R
from ....schema.proactive_agent import ManageProactiveAgentRequest


@R.register("manage_proactive_agent_step")
class ManageProactiveAgentStep(BaseAgentStep):
    """Validate management arguments before accessing configuration or sessions."""

    @staticmethod
    def _settings(config):
        """Select task settings without internal batching or duplicate identities."""
        keys = ["trigger_type", "enabled", "temporary"]
        keys.append("cron" if config["trigger_type"] == "cron" else "min_trigger_interval_seconds")
        if config["notify_window"] is not None:
            keys.append("notify_window")
        return {key: config[key] for key in keys}

    @classmethod
    def _status(cls, value):
        """Keep execution facts and the original memory path, without memory previews."""
        result = {
            "name": value["agent_id"],
            **cls._settings(value["config"]),
            **{
                key: value[key]
                for key in (
                    "memory_path",
                    "running",
                    "pending_events",
                    "in_flight_events",
                    "last_completed_at",
                )
            },
        }
        for key in ("next_trigger_at", "error"):
            if value[key] is not None:
                result[key] = value[key]
        if value["running"] and value["last_started_at"] is not None:
            result["last_started_at"] = value["last_started_at"]
        return result

    async def execute(self):
        runtime = self.agent_runtime
        request = ManageProactiveAgentRequest.model_validate(self.parameters())
        self.require_paths(runtime.agents.directory)
        if request.action in {"create", "update", "query"}:
            self.require_paths(self.workspace_path)
        settings = request.model_dump(
            exclude_unset=True,
            exclude={"action", "name", "instructions", "question"},
        )
        if request.action == "list":
            listing = await runtime.list_agents()
            answer = {
                "items": [
                    {
                        key: item["config"][key]
                        for key in ("name", "trigger_type", "enabled", "cron")
                        if item["config"][key] is not None
                    }
                    for item in listing["items"]
                ],
            }
        elif request.action == "create":
            value = await runtime.create({"name": request.name, "description": request.instructions, **settings})
            answer = {"name": value["agent_id"], "memory_path": value["memory_path"]}
            if value["next_trigger_at"] is not None:
                answer["next_trigger_at"] = value["next_trigger_at"]
            if value["restored"]:
                answer["restored"] = True
        elif request.action == "status":
            answer = self._status(await runtime.status(request.name))
        elif request.action == "update":
            if settings.get("trigger_type") == "event":
                settings.setdefault("cron", None)
            elif settings.get("trigger_type") == "cron":
                settings.setdefault("min_trigger_interval_seconds", 0)
            value = await runtime.update(request.name, settings)
            answer = {
                "name": value["agent_id"],
                "updated": {key: value["config"][key] for key in settings},
            }
            if value["next_trigger_at"] is not None:
                answer["next_trigger_at"] = value["next_trigger_at"]
        elif request.action == "delete":
            await runtime.delete(request.name)
            answer = f"Deleted task '{request.name}'; history files retained."
        else:
            response = await self.run_job(
                self.kwargs.get("query_job", "proactive_query"),
                agent_id=request.name,
                query=request.question,
                _allowed_paths=self.context.get("_allowed_paths"),
            )
            self.context.response.success = response.success
            answer = response.answer["answer"] if response.success else response.answer
        self.context.response.answer = answer
        return self.context.response
