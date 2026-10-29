"""ReMe agent wrapper with Home's bounded investigation runtime."""

from typing import Any

from reme.components.agent_wrapper import AsAgentWrapper

from .bounded import BoundedReActAgent, RunTimeBudgetTimer


class BoundedAgentWrapper(AsAgentWrapper):
    """Reuse ReMe Job tools, injected arguments, permissions and persisted sessions.

    ``react_config.max_iters`` configures the iteration cap. Optional
    ``time_budget_minutes`` adds advisory reminders; each built SDK agent owns
    its own timer so requests never share budget state.
    """

    def _create_agent(self, *, options: dict[str, Any], **kwargs: Any) -> BoundedReActAgent:
        budget = options.get("time_budget_minutes")
        timer = None if budget is None else RunTimeBudgetTimer(budget, agent_label=self.name)
        return BoundedReActAgent(run_time_budget_timer=timer, **kwargs)
