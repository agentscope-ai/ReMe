"""ReMe agent wrapper with Home's bounded investigation runtime."""

from typing import Any

from reme.components.agent_wrapper import AsAgentWrapper
from reme.components.as_llm import BaseAsLLM

from ..judge import HomeJudge
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


class CaseStudyAgentWrapper(BoundedAgentWrapper):
    """Validate Home's host configuration before requiring the investigation model.

    A configured model and the host judge start before this wrapper. Optional
    model binding preserves Home's missing-judge diagnostic; startup still
    rejects an absent investigation model after the judge is validated.
    """

    def __init__(self, as_llm: str = "judge", **kwargs):
        super().__init__(as_llm="", **kwargs)
        self.home_judge = self.bind("default", HomeJudge, optional=False)
        self.as_llm = self.bind(as_llm, BaseAsLLM, optional=True)

    async def _start(self):
        await super()._start()
        if self.as_llm is None:
            raise ValueError("Case-study agent requires a configured as_llm component")
