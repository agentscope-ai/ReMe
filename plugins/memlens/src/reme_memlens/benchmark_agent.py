"""Benchmark-local compatibility with the pre-AgentReply ReMe evaluation."""

from agentscope.agent import Agent
from agentscope.message import Msg

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.schema import AgentReply


class BenchmarkAsAgentWrapper(AsAgentWrapper):
    """Accept error-free, unstructured iteration-limit results in this benchmark."""

    def __init__(self, **kwargs):
        # The alias selects this class; native image capability remains AgentScope.
        super().__init__(**{**kwargs, "backend": "agentscope"})

    async def _finish_reply(self, agent: Agent, last_msg: Msg, kwargs: dict) -> AgentReply:
        result = await super()._finish_reply(agent, last_msg, kwargs)
        message = result["raw"]["message"]
        if (
            message["finished_reason"] == "exceed_max_iters"
            and message.get("error") is None
            and kwargs.get("output_schema") is None
        ):
            result["status"] = "completed"
            result["error"] = None
        return result
