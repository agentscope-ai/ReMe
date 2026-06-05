"""AgentScope backend for the unified agent wrapper."""

from typing import Any, TYPE_CHECKING

from agentscope.agent import Agent
from agentscope.message import TextBlock, ToolResultState, UserMsg
from agentscope.tool import FunctionTool, ToolChunk, Toolkit

from .base_agent_wrapper import BaseAgentWrapper
from ..as_llm import BaseAsLLM
from ..component_registry import R

if TYPE_CHECKING:
    from ..job.base_job import BaseJob


@R.register("agentscope")
class AsAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by AgentScope framework.

    Args:
        as_llm: Name of the bound as_llm component (resolved via app_context).
    Kwargs:
        system_prompt: System prompt for the agent.
        tools: list[BaseJob] to register as agent wrapper tools.
    """

    def __init__(self, as_llm: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.as_llm = self.bind(as_llm, BaseAsLLM, optional=False)

    @staticmethod
    def _make_tool(job: "BaseJob") -> FunctionTool:
        async def run_job(**kwargs) -> ToolChunk:
            response = await job(**kwargs)
            return ToolChunk(
                content=[TextBlock(text=str(response.answer))],
                state=ToolResultState.SUCCESS if response.success else ToolResultState.ERROR,
            )

        tool = FunctionTool(func=run_job, name=job.name, description=job.description)
        if job.parameters:
            tool.input_schema = job.parameters
        return tool

    async def reply(self, inputs: Any, session_id: str | None = None, **kwargs) -> tuple[str, Any]:
        model = self.as_llm.model if self.as_llm else None
        if model is None:
            raise ValueError("AsAgentWrapper requires a bound as_llm component with a valid model.")

        tools: list["BaseJob"] = self.kwargs.get("tools", [])
        toolkit = Toolkit(tools=[self._make_tool(job) for job in tools]) if tools else Toolkit()

        agent = Agent(
            name=self.name,
            system_prompt=self.kwargs.get("system_prompt", "You are a helpful assistant."),
            model=model,
            toolkit=toolkit,
        )

        if isinstance(inputs, str):
            inputs = UserMsg(name="user", content=inputs)
        await agent.observe(inputs)
        await agent.reply()

        # save
        return agent.state.session_id, agent.state.context[-1]
