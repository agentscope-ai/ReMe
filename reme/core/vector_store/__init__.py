"""vector store"""

from .base_vector_store import BaseVectorStore
from .chroma_vector_store import ChromaVectorStore
from .es_vector_store import ESVectorStore
from .hologres_store import HologresVectorStore
from .local_vector_store import LocalVectorStore
from .pgvector_store import PGVectorStore
from .qdrant_vector_store import QdrantVectorStore
from ..registry_factory import R

__all__ = [
    "BaseVectorStore",
    "ChromaVectorStore",
    "ESVectorStore",
    "HologresVectorStore",
    "LocalVectorStore",
    "PGVectorStore",
    "QdrantVectorStore",
]

R.vector_stores.register("chroma")(ChromaVectorStore)
R.vector_stores.register("es")(ESVectorStore)
R.vector_stores.register("hologres")(HologresVectorStore)
R.vector_stores.register("local")(LocalVectorStore)
R.vector_stores.register("pgvector")(PGVectorStore)
R.vector_stores.register("qdrant")(QdrantVectorStore)

try:
    from .obvec_vector_store import ObVecVectorStore

    R.vector_stores.register("obvec")(ObVecVectorStore)
    __all__.append("ObVecVectorStore")
except ImportError:
    pass

try:
    from .zvec_vector_store import ZvecVectorStore

    R.vector_stores.register("zvec")(ZvecVectorStore)
    __all__.append("ZvecVectorStore")
except ImportError:
    pass

try:
    from .seekdb_vector_store import SeekdbVectorStore

    R.vector_stores.register("seekdb")(SeekdbVectorStore)
    __all__.append("SeekdbVectorStore")
except ImportError:
    pass
