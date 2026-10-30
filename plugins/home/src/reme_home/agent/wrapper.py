"""ReMe agent wrapper with Home's bounded investigation runtime."""

import json
from typing import Any

from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import FunctionTool, ToolChunk

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

    def __init__(self, as_llm: str = "home_case_study", **kwargs):
        super().__init__(as_llm="", **kwargs)
        self.home_judge = self.bind("default", HomeJudge, optional=False)
        self.as_llm = self.bind(as_llm, BaseAsLLM, optional=True)

    async def _start(self):
        await super()._start()
        if self.as_llm is None:
            raise ValueError("Case-study agent requires a configured as_llm component")

    @classmethod
    def _make_tool(cls, job, tool_context_id=None, injected_job_kwargs=None):
        """Expose structured investigation evidence as JSON without changing core Job tools."""
        injected = cls._resolve_injected_job_kwargs(
            {"tool_context_id": tool_context_id, "injected_job_kwargs": injected_job_kwargs},
        )

        async def run_job(**kwargs):
            response = await job(**cls._merge_injected_job_kwargs(kwargs, injected))
            answer = response.answer
            text = json.dumps(answer, ensure_ascii=False) if isinstance(answer, (dict, list)) else str(answer)
            state = ToolResultState.SUCCESS if response.success else ToolResultState.ERROR
            return ToolChunk(content=[TextBlock(text=text)], state=state)

        tool = FunctionTool(func=run_job, name=job.name, description=job.description, is_concurrency_safe=False)
        if parameters := cls._strip_injected_parameters(job.parameters, injected):
            tool.input_schema = parameters
        return tool
