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
        keys = ["agent_type", "enabled", "temporary"]
        keys.append("cron" if config["agent_type"] == "cron" else "min_trigger_interval_seconds")
        if config["notify_window"] is not None:
            keys.append("notify_window")
        return {key: config[key] for key in keys}

    @classmethod
    def _status(cls, value):
        """Keep execution facts and the original memory path, without memory previews."""
        result = {
            "agent_name": value["agent_name"],
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
            exclude={"action", "agent_name", "task_memory", "question"},
        )
        if request.action == "list":
            listing = await runtime.list_agents()
            answer = {
                "items": [
                    {
                        **{
                            key: item["config"][key]
                            for key in ("agent_name", "agent_type", "enabled", "cron")
                            if item["config"][key] is not None
                        },
                        "memory_path": item["memory_path"],
                    }
                    for item in listing["items"]
                ],
            }
        elif request.action == "create":
            value = await runtime.create(
                {"agent_name": request.agent_name, "task_memory": request.task_memory, **settings},
            )
            answer = {"agent_name": value["agent_name"], "memory_path": value["memory_path"]}
            if value["next_trigger_at"] is not None:
                answer["next_trigger_at"] = value["next_trigger_at"]
            if value["restored"]:
                answer["restored"] = True
        elif request.action == "status":
            answer = self._status(await runtime.status(request.agent_name))
        elif request.action == "update":
            if settings.get("agent_type") == "proactive":
                settings.setdefault("cron", None)
            elif settings.get("agent_type") == "cron":
                settings.setdefault("min_trigger_interval_seconds", 0)
            value = await runtime.update(request.agent_name, settings)
            answer = {
                "agent_name": value["agent_name"],
                "updated": {key: value["config"][key] for key in settings},
            }
            if value["next_trigger_at"] is not None:
                answer["next_trigger_at"] = value["next_trigger_at"]
        elif request.action == "delete":
            await runtime.delete(request.agent_name)
            answer = f"Deleted task '{request.agent_name}'; history files retained."
        else:
            response = await self.run_job(
                self.kwargs.get("query_job", "proactive_query"),
                agent_name=request.agent_name,
                query=request.question,
                _allowed_paths=self.context.get("_allowed_paths"),
            )
            self.context.response.success = response.success
            answer = response.answer["answer"] if response.success else response.answer
        self.context.response.answer = answer
        return self.context.response
