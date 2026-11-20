"""MoonshotAsLLM provider backend."""

from agentscope.credential import MoonshotCredential

from .base_as_llm import BaseAsLLM
from ..component_registry import R


@R.register("moonshot")
class MoonshotAsLLM(BaseAsLLM):
    """Moonshot chat model wrapper."""

    credential_cls = MoonshotCredential
