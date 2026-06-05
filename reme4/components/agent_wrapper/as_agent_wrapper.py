"""AgentScope backend for the unified agent wrapper."""

from typing import Any, TYPE_CHECKING

from agentscope.agent import Agent, ContextConfig, ModelConfig, ReActConfig
from agentscope.message import TextBlock, ToolResultState, UserMsg, SystemMsg
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, ToolChunk, Toolkit

from .base_agent_wrapper import BaseAgentWrapper
from ..as_llm import BaseAsLLM
from ..component_registry import R

if TYPE_CHECKING:
    from ..job.base_job import BaseJob


@R.register("agentscope")
class AsAgentWrapper(BaseAgentWrapper):
    """Agent wrapper backed by AgentScope framework."""

    def __init__(self, as_llm: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.as_llm = self.bind(as_llm, BaseAsLLM, optional=False)

    @staticmethod
    def _make_tool(job: "BaseJob") -> FunctionTool:
        async def run_job(**kwargs) -> ToolChunk:
            response = await job(**kwargs)
            state = ToolResultState.SUCCESS if response.success else ToolResultState.ERROR
            return ToolChunk(content=[TextBlock(text=str(response.answer))], state=state)

        tool = FunctionTool(func=run_job, name=job.name, description=job.description)
        if job.parameters:
            tool.input_schema = job.parameters
        return tool

    async def reply(self, inputs: Any, session_id: str | None = None, **kwargs) -> tuple[str, Any]:
        model = self.as_llm.model if self.as_llm else None
        if model is None:
            raise ValueError("AsAgentWrapper requires a bound as_llm component with a valid model.")

        for k, v in self.kwargs.items():
            kwargs.setdefault(k, v)

        system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
        tools: list["BaseJob"] = kwargs.get("tools", [])
        toolkit = Toolkit(tools=[self._make_tool(job) for job in tools]) if tools else Toolkit()

        perm_mode = PermissionMode(kwargs.get("permission_mode", "bypass"))
        state = AgentState(permission_context=PermissionContext(mode=perm_mode))

        agent = Agent(
            name=self.name,
            system_prompt=system_prompt,
            model=model,
            toolkit=toolkit,
            state=state,
            model_config=ModelConfig(**(kwargs.get("model_config") or {})),
            context_config=ContextConfig(**(kwargs.get("context_config") or {})),
            react_config=ReActConfig(**(kwargs.get("react_config") or {})),
        )

        if isinstance(inputs, str):
            inputs = UserMsg(name="user", content=inputs)

        if output_schema := kwargs.get("output_schema"):
            messages = [SystemMsg(name="system", content=system_prompt), inputs]
            assert isinstance(output_schema, dict),  "Output schema must be a dict."
            res = await model.generate_structured_output(messages=messages, structured_model=output_schema)
            return agent.state.session_id, res.content

        await agent.observe(inputs)
        await agent.reply()
        return agent.state.session_id, agent.state.context[-1]
