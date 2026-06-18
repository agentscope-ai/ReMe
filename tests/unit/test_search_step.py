"""Unit tests for SearchStep without embedding or LLM dependencies."""

import asyncio

from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import LinkScopeEnum
from reme.schema import FileChunk, FileLink, FileNode
from reme.steps.index import SearchStep


class FakeSearchStore(BaseFileStore):
    """Minimal file_store for SearchStep: static search results and empty graph links."""

    def __init__(
        self,
        vector_results: list[FileChunk] | None = None,
        keyword_results: list[FileChunk] | None = None,
    ):
        super().__init__(name="fake_search_store")
        self.vector_results = vector_results or []
        self.keyword_results = keyword_results or []
        self.calls: list[tuple[str, str, int, dict]] = []

    async def upsert(self, files: list[tuple[FileNode, list[FileChunk]]]) -> None:
        raise NotImplementedError

    async def delete(self, path: str | list[str]) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError

    async def get_nodes(self, paths: list[str] | None = None) -> list[FileNode]:
        return []

    async def get_outlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        return []

    async def get_inlinks(
        self,
        path: str,
        scope: LinkScopeEnum = LinkScopeEnum.REAL,
    ) -> list[FileLink]:
        return []

    async def vector_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        self.calls.append(("vector", query, limit, search_filter))
        return self.vector_results[:limit]

    async def keyword_search(self, query: str, limit: int, search_filter: dict) -> list[FileChunk]:
        self.calls.append(("keyword", query, limit, search_filter))
        return self.keyword_results[:limit]


def _chunk(
    chunk_id: str,
    path: str,
    text: str,
    score_key: str,
    score: float,
    line: int = 1,
) -> FileChunk:
    return FileChunk(
        id=chunk_id,
        path=path,
        text=text,
        start_line=line,
        end_line=line,
        scores={score_key: score, "score": score},
    )


def test_search_step_rrf_merges_vector_and_keyword_by_chunk_id():
    """Hybrid search fuses same-id hits once and keeps per-branch scores in metadata."""

    async def run():
        shared_v = _chunk("shared", "daily/a.md", "shared vector text", "vector", 0.92, line=3)
        vector_only = _chunk("vector-only", "daily/b.md", "vector text", "vector", 0.71)
        keyword_only = _chunk("keyword-only", "digest/c.md", "keyword text", "keyword", 8.0)
        shared_k = _chunk("shared", "daily/a.md", "shared keyword text", "keyword", 7.0, line=3)
        store = FakeSearchStore(
            vector_results=[shared_v, vector_only],
            keyword_results=[keyword_only, shared_k],
        )
        step = SearchStep(file_store=store, vector_weight=0.5, candidate_multiplier=2, expand_links=False)
        ctx = RuntimeContext(query="alpha", limit=3, search_filter={"path_prefix": "daily/"})

        resp = await step(ctx)

        assert resp.success is True
        assert resp.metadata["counts"] == {"vector": 2, "keyword": 2, "returned": 3, "hybrid": True}
        assert [r["id"] for r in resp.metadata["results"]] == ["shared", "keyword-only", "vector-only"]
        shared = resp.metadata["results"][0]
        assert shared["scores"]["vector"] == 0.92
        assert shared["scores"]["keyword"] == 7.0
        assert shared["scores"]["score"] > resp.metadata["results"][1]["scores"]["score"]
        assert "daily/a.md:3-3" in resp.answer
        assert "vector=0.9200" in resp.answer
        assert "keyword=7.0000" in resp.answer
        assert {call[0] for call in store.calls} == {"vector", "keyword"}
        assert all(call[2] == 6 for call in store.calls)
        assert all(call[3] == {"path_prefix": "daily/"} for call in store.calls)

    asyncio.run(run())


def test_search_step_keyword_only_uses_keyword_scores_and_min_score():
    """When vector has no hits, SearchStep returns keyword results directly and applies min_score."""

    async def run():
        high = _chunk("high", "daily/high.md", "strong keyword hit", "keyword", 4.0)
        low = _chunk("low", "daily/low.md", "weak keyword hit", "keyword", 0.2)
        store = FakeSearchStore(keyword_results=[high, low])
        step = SearchStep(file_store=store, expand_links=False)
        ctx = RuntimeContext(query="keyword", limit=5, min_score=1.0)

        resp = await step(ctx)

        assert resp.metadata["counts"] == {"vector": 0, "keyword": 2, "returned": 1, "hybrid": False}
        assert [r["id"] for r in resp.metadata["results"]] == ["high"]
        assert "keyword=4.0000" not in resp.answer
        assert "score=4.0000" in resp.answer
        assert "daily/low.md" not in resp.answer

    asyncio.run(run())


def test_search_step_empty_query_fails_before_store_calls():
    """Empty queries fail fast and do not call file_store search methods."""

    async def run():
        store = FakeSearchStore()
        step = SearchStep(file_store=store)
        resp = await step(RuntimeContext(query="  ", limit=5))

        assert resp.success is False
        assert resp.answer == "Error: query cannot be empty"
        assert not store.calls

    asyncio.run(run())
