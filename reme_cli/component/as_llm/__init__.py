"""Module for registering AgentScope LLM models."""

from agentscope.model import OpenAIChatModel

from ..registry_factory import R

R.as_llms.register("openai")(OpenAIChatModel)
