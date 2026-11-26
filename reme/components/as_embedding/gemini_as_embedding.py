"""GeminiAsEmbedding provider backend."""

from agentscope.credential import GeminiCredential

from .base_as_embedding import BaseAsEmbedding
from ...core.registry import R


@R.register("gemini")
class GeminiAsEmbedding(BaseAsEmbedding):
    """Gemini embedding model wrapper."""

    credential_cls = GeminiCredential
