"""Module for registering AgentScope token counters."""

from agentscope.token import OpenAITokenCounter
from agentscope.token import HuggingFaceTokenCounter

from ..registry_factory import R

R.as_token_counters.register("openai")(OpenAITokenCounter)
R.as_token_counters.register("hf")(HuggingFaceTokenCounter)
