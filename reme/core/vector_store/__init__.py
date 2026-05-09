"""vector store"""

from .base_vector_store import BaseVectorStore
from .chroma_vector_store import ChromaVectorStore
from .es_vector_store import ESVectorStore
from .hologres_store import HologresVectorStore
from .local_vector_store import LocalVectorStore
from .obvec_vector_store import ObVecVectorStore
from .pgvector_store import PGVectorStore
from .qdrant_vector_store import QdrantVectorStore
from .zvec_vector_store import ZvecVectorStore
from ..registry_factory import R

__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore",
    "ESVectorStore",
    "HologresVectorStore",
    "LocalVectorStore",
    "ObVecVectorStore",
    "PGVectorStore",
    "QdrantVectorStore",
    "ZvecVectorStore",
]

R.vector_stores.register("chroma")(ChromaVectorStore)
R.vector_stores.register("es")(ESVectorStore)
R.vector_stores.register("hologres")(HologresVectorStore)
R.vector_stores.register("local")(LocalVectorStore)
R.vector_stores.register("obvec")(ObVecVectorStore)
R.vector_stores.register("pgvector")(PGVectorStore)
R.vector_stores.register("qdrant")(QdrantVectorStore)
R.vector_stores.register("zvec")(ZvecVectorStore)
