"""Module for registering AgentScope LLM formatters."""

from agentscope.formatter import DashScopeChatFormatter
from agentscope.formatter import OpenAIChatFormatter

from ..registry_factory import R

R.as_llm_formatters.register("openai")(OpenAIChatFormatter)
R.as_llm_formatters.register("dashscope")(DashScopeChatFormatter)
