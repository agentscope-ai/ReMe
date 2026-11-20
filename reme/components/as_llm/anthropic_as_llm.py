"""AnthropicAsLLM provider backend."""

from agentscope.credential import AnthropicCredential

from .base_as_llm import BaseAsLLM
from ...core.registry import R


@R.register("anthropic")
class AnthropicAsLLM(BaseAsLLM):
    """Anthropic chat model wrapper."""

    credential_cls = AnthropicCredential
