"""XAIAsLLM provider backend."""

from agentscope.credential import XAICredential

from .base_as_llm import BaseAsLLM
from ..component_registry import R


@R.register("xai")
class XAIAsLLM(BaseAsLLM):
    """xAI chat model wrapper."""

    credential_cls = XAICredential
