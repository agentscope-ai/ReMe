"""OllamaAsEmbedding provider backend."""

from agentscope.credential import OllamaCredential

from .base_as_embedding import BaseAsEmbedding
from ..component_registry import R


@R.register("ollama")
class OllamaAsEmbedding(BaseAsEmbedding):
    """Ollama embedding model wrapper."""

    credential_cls = OllamaCredential
    endpoint_fields = ("host",)
    endpoint_env = "OLLAMA_HOST"
    default_endpoint = "http://127.0.0.1:11434"
