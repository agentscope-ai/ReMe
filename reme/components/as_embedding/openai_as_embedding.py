"""OpenAIAsEmbedding provider backend."""

from agentscope.credential import OpenAICredential

from .base_as_embedding import BaseAsEmbedding
from ..component_registry import R


@R.register("openai")
class OpenAIAsEmbedding(BaseAsEmbedding):
    """OpenAI embedding model wrapper."""

    credential_cls = OpenAICredential
    endpoint_fields = ("base_url",)
    endpoint_env = "OPENAI_BASE_URL"
    default_endpoint = "https://api.openai.com/v1"

    def _model_endpoint(self) -> str:
        """Read the endpoint the OpenAI client actually resolved."""
        assert self.model is not None
        client = getattr(self.model, "client", None)
        base_url = getattr(client, "base_url", None)
        return str(base_url).rstrip("/") if base_url is not None else super()._model_endpoint()
