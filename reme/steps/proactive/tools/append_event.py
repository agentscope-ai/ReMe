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
            set(arguments) != {"agent_name", "events"}
            or not isinstance(arguments["events"], list)
            or not arguments["events"]
        ):
            raise ValueError("append_agent_events requires agent_name and a nonempty events array")
        runtime = self.agent_runtime
        self.require_paths(runtime.agents.directory)
        events = [Event.model_validate(event) for event in arguments["events"]]
        for event in events:
            self.require_paths(*(runtime.files.path(path) for path in event.paths))
        self.context.response.answer = await runtime.append(arguments["agent_name"], events)
        return self.context.response
