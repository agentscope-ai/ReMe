"""DeepSeekAsLLM provider backend."""

from agentscope.credential import DeepSeekCredential

from .base_as_llm import BaseAsLLM
from ..component_registry import R


@R.register("deepseek")
class DeepSeekAsLLM(BaseAsLLM):
    """DeepSeek chat model wrapper."""

    credential_cls = DeepSeekCredential
