"""vector store"""

from .base_vector_store import BaseVectorStore
from .chroma_vector_store import ChromaVectorStore
from .es_vector_store import ESVectorStore
from .local_vector_store import LocalVectorStore
from .pgvector_store import PGVectorStore
from .qdrant_vector_store import QdrantVectorStore
from ..registry_factory import R

__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore",
    "ESVectorStore",
    "LocalVectorStore",
    "PGVectorStore",
    "QdrantVectorStore",
]

R.vector_stores.register("chroma")(ChromaVectorStore)
R.vector_stores.register("es")(ESVectorStore)
R.vector_stores.register("local")(LocalVectorStore)
R.vector_stores.register("pgvector")(PGVectorStore)
R.vector_stores.register("qdrant")(QdrantVectorStore)
