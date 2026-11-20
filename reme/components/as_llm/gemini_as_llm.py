"""GeminiAsLLM provider backend."""

from agentscope.credential import GeminiCredential

from .base_as_llm import BaseAsLLM
from ..component_registry import R


@R.register("gemini")
class GeminiAsLLM(BaseAsLLM):
    """Gemini chat model wrapper."""

    credential_cls = GeminiCredential
