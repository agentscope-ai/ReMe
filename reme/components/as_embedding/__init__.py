"""AgentScope embedding components."""

from .base_as_embedding import BaseAsEmbedding
from .openai_as_embedding import OpenAIAsEmbedding
from .dashscope_as_embedding import DashScopeAsEmbedding
from .dashscope_multimodal_as_embedding import DashScopeMultiModalAsEmbedding
from .gemini_as_embedding import GeminiAsEmbedding
from .ollama_as_embedding import OllamaAsEmbedding

__all__ = [
    "BaseAsEmbedding",
    "OpenAIAsEmbedding",
    "DashScopeAsEmbedding",
    "DashScopeMultiModalAsEmbedding",
    "GeminiAsEmbedding",
    "OllamaAsEmbedding",
]
