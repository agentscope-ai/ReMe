"""Deliver observations to an agent without exposing its internal stream name."""

from ..agent import BaseAgentStep
from ....components import R
from ....schema.event import Event


@R.register("append_event_step")
class AppendEventStep(BaseAgentStep):
    """Validate complete events and every referenced workspace path before append."""

    async def execute(self):
        arguments = self.parameters()
        if (
            set(arguments) != {"agent_id", "events"}
            or not isinstance(arguments["events"], list)
            or not arguments["events"]
        ):
            raise ValueError("append_event requires agent_id and a nonempty events array")
        runtime = self.agent_runtime
        self.require_paths(runtime.agents.directory)
        if (await runtime.definition(arguments["agent_id"])).trigger_type != "event":
            raise ValueError("external observations require an event agent")
        events = [Event.model_validate(event) for event in arguments["events"]]
        for event in events:
            if event.source == "cron":
                raise ValueError("cron is a reserved framework source")
            self.require_paths(*(runtime.files.path(path) for path in event.paths))
        self.context.response.answer = await runtime.append(arguments["agent_id"], events)
        return self.context.response
