"""Module for registering AgentScope LLM formatters."""

from agentscope.formatter import DashScopeChatFormatter

from .reme_openai_chat_formatter import ReMeOpenAIChatFormatter
from ..registry_factory import R

R.as_llm_formatters.register("openai")(ReMeOpenAIChatFormatter)
R.as_llm_formatters.register("dashscope")(DashScopeChatFormatter)
