"""DashScopeMultiModalAsEmbedding provider backend."""

from agentscope.credential import DashScopeCredential

from .base_as_embedding import BaseAsEmbedding
from ..component_registry import R


@R.register("dashscope_multimodal")
class DashScopeMultiModalAsEmbedding(BaseAsEmbedding):
    """DashScope multimodal embedding model wrapper."""

    credential_cls = DashScopeCredential
