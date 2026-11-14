"""DashScopeAsLLM provider backend."""

from agentscope.credential import DashScopeCredential

from .base_as_llm import BaseAsLLM
from ..component_registry import R


@R.register("dashscope")
class DashScopeAsLLM(BaseAsLLM):
    """DashScope chat model wrapper."""

    credential_cls = DashScopeCredential
