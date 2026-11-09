"""Benchmark-local compatibility with the pre-AgentReply ReMe evaluation."""

from agentscope.agent import Agent
from agentscope.message import Msg

from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.schema import AgentReply


class BenchmarkAsAgentWrapper(AsAgentWrapper):
    """Accept the legacy iteration-limit outcome only in this benchmark config.

    Structured-output calls retain the strict core outcome: these benchmark
    memory/answer paths do not request an extra structured-model response.
    """

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
            result["raw"]["benchmark_compat"] = {
                "policy": "agentscope-exceed-max-iters-legacy-v1",
                "legacy_source_commit": "d3e79918622173306be79ecda6494acadc7c4312",
                "native_status": result["status"],
                "native_error": result["error"],
                "native_termination_reason": message["finished_reason"],
            }
            result["status"] = "completed"
            result["error"] = None
        return result
