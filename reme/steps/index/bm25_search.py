"""Plain BM25 keyword search with tool-context deduplication."""

from ._single_search import _SingleSearchStep
from ...components import R
from ...schema import FileChunk


@R.register("bm25_search_step")
class Bm25SearchStep(_SingleSearchStep):
    """Search the BM25 keyword backend using the shared result contract."""

    async def _retrieve(self, query: str, candidates: int) -> list[FileChunk]:
        return await self.file_store.keyword_search(query, candidates, {})
