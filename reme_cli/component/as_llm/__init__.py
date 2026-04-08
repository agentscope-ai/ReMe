"""Module for registering AgentScope LLM models."""

from agentscope.model import DashScopeChatModel
from agentscope.model import OpenAIChatModel

from ..registry_factory import R

R.as_llms.register("openai")(OpenAIChatModel)
R.as_llms.register("dashscope")(DashScopeChatModel)
