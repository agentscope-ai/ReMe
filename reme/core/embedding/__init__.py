"""embedding"""

from .base_embedding_model import BaseEmbeddingModel
from .openai_embedding_model import OpenAIEmbeddingModel
from .openai_embedding_model_sync import OpenAIEmbeddingModelSync
from ..registry_factory import R

__all__ = [
    "BaseEmbeddingModel",
    "OpenAIEmbeddingModel",
    "OpenAIEmbeddingModelSync",
]

R.embedding_models.register("openai")(OpenAIEmbeddingModel)
R.embedding_models.register("openai_sync")(OpenAIEmbeddingModelSync)
