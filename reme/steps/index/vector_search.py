"""Plain vector search with tool-context deduplication."""

from ._single_search import _SingleSearchStep
from ...components import R
from ...schema import FileChunk


@R.register("vector_search_step")
class VectorSearchStep(_SingleSearchStep):
    """Search the vector backend using the shared result contract."""

    async def _retrieve(self, query: str, candidates: int) -> list[FileChunk]:
        return await self.file_store.vector_search(query, candidates, {})
