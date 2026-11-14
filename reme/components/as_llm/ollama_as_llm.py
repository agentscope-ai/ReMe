"""OllamaAsLLM provider backend."""

from agentscope.credential import OllamaCredential

from .base_as_llm import BaseAsLLM
from ..component_registry import R


@R.register("ollama")
class OllamaAsLLM(BaseAsLLM):
    """Ollama chat model wrapper."""

    credential_cls = OllamaCredential
