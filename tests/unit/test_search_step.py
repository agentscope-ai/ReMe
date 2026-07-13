"""Unit tests for SearchStep without embedding or LLM dependencies."""

import asyncio

from reme.components.file_store import BaseFileStore
from reme.components import ApplicationContext
from reme.components.runtime_context import RuntimeContext
from reme.enumeration import LinkScopeEnum
from reme.schema import FileChunk, FileLink, FileNode
from reme.steps.index import AddDraftStep, Bm25SearchStep, ReadAllDraftStep, SearchStep, VectorSearchStep


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


def test_draft_steps_accumulate_by_tool_context_id():
    """Drafts are stored in app metadata and isolated by injected tool_context_id."""

    async def run():
        app_context = ApplicationContext()
        add = AddDraftStep(app_context=app_context)
        read = ReadAllDraftStep(app_context=app_context)

        await add(RuntimeContext(text="first", tool_context_id="ctx-1"))
        await add(RuntimeContext(text="second", tool_context_id="ctx-1"))
        await add(RuntimeContext(text="other", tool_context_id="ctx-2"))

        resp = await read(RuntimeContext(tool_context_id="ctx-1"))

        assert resp.answer == "first\nsecond"
        assert resp.metadata["draft_count"] == 2

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


def test_plain_search_steps_apply_min_score_before_truncation():
    """Vector-only and BM25-only tools should not return hits below ``min_score``."""

    async def run():
        vector_store = FakeSearchStore(
            vector_results=[
                _chunk("vector-high", "daily/high.md", "strong vector hit", "vector", 0.9),
                _chunk("vector-low", "daily/low.md", "weak vector hit", "vector", 0.2),
            ],
        )
        keyword_store = FakeSearchStore(
            keyword_results=[
                _chunk("keyword-high", "daily/high.md", "strong keyword hit", "keyword", 4.0),
                _chunk("keyword-low", "daily/low.md", "weak keyword hit", "keyword", 0.2),
            ],
        )

        vector = await VectorSearchStep(file_store=vector_store)(
            RuntimeContext(query="alpha", limit=5, min_score=0.5),
        )
        keyword = await Bm25SearchStep(file_store=keyword_store)(
            RuntimeContext(query="alpha", limit=5, min_score=1.0),
        )

        assert [result["id"] for result in vector.metadata["results"]] == ["vector-high"]
        assert [result["id"] for result in keyword.metadata["results"]] == ["keyword-high"]
        assert vector.answer == "strong vector hit"
        assert keyword.answer == "strong keyword hit"

    asyncio.run(run())


def test_search_step_tool_context_deduplicates_returned_chunks_only():
    """When tool_context_id is supplied, repeated searches skip previously returned chunks."""

    async def run():
        chunks = [
            _chunk("a", "daily/a.md", "first", "keyword", 5.0),
            _chunk("b", "daily/b.md", "second", "keyword", 4.0),
            _chunk("c", "daily/c.md", "third", "keyword", 3.0),
        ]
        store = FakeSearchStore(keyword_results=chunks)
        step = SearchStep(file_store=store, expand_links=False)

        first = await step(RuntimeContext(query="alpha", limit=2, tool_context_id="ctx-1"))
        second = await step(RuntimeContext(query="alpha", limit=2, tool_context_id="ctx-1"))
        third = await step(RuntimeContext(query="alpha", limit=2))

        assert [r["id"] for r in first.metadata["results"]] == ["a", "b"]
        assert first.metadata["dedup"] == {
            "tool_context_id": "ctx-1",
            "seen_before": 0,
            "skipped_seen": 0,
            "seen_after": 2,
            "expired": 0,
            "ttl_seconds": 86400.0,
        }
        assert [r["id"] for r in second.metadata["results"]] == ["c"]
        assert second.metadata["dedup"]["seen_before"] == 2
        assert second.metadata["dedup"]["skipped_seen"] == 2
        assert second.metadata["dedup"]["seen_after"] == 3
        assert [r["id"] for r in third.metadata["results"]] == ["a", "b"]
        assert "dedup" not in third.metadata

    asyncio.run(run())


def test_search_step_passes_metadata_filter_to_store():
    """Search filters can target chunk metadata such as conversation_date."""

    async def run():
        hit = _chunk("hit", "daily/2023-01-19/event.md", "historical hit", "keyword", 3.0)
        store = FakeSearchStore(keyword_results=[hit])
        step = SearchStep(file_store=store, expand_links=False)
        search_filter = {"metadata": {"conversation_date": "2023-01-19"}}

        resp = await step(RuntimeContext(query="Jon job", limit=5, search_filter=search_filter))

        assert resp.success is True
        assert resp.metadata["results"][0]["id"] == "hit"
        assert all(call[3] == search_filter for call in store.calls)

    asyncio.run(run())


def test_search_step_tool_context_seen_chunks_expire_after_ttl():
    """Seen chunk ids under a tool_context_id are reusable after the configured TTL."""

    async def run():
        now = 1000.0
        chunks = [
            _chunk("a", "daily/a.md", "first", "keyword", 5.0),
            _chunk("b", "daily/b.md", "second", "keyword", 4.0),
        ]
        store = FakeSearchStore(keyword_results=chunks)
        step = SearchStep(
            file_store=store,
            expand_links=False,
            seen_ttl_hours=1,
            clock=lambda: now,
        )

        first = await step(RuntimeContext(query="alpha", limit=1, tool_context_id="ctx-1"))
        now = 4601.0
        second = await step(RuntimeContext(query="alpha", limit=1, tool_context_id="ctx-1"))

        assert [r["id"] for r in first.metadata["results"]] == ["a"]
        assert [r["id"] for r in second.metadata["results"]] == ["a"]
        assert second.metadata["dedup"]["expired"] == 1
        assert second.metadata["dedup"]["seen_before"] == 0
        assert second.metadata["dedup"]["ttl_seconds"] == 3600.0

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


def test_search_step_start_end_date_promoted_into_search_filter():
    """start_date and end_date from context are promoted into search_filter passed to store."""

    async def run():
        hit = _chunk("hit", "daily/a.md", "some text", "keyword", 5.0)
        store = FakeSearchStore(keyword_results=[hit])
        step = SearchStep(file_store=store, expand_links=False)
        ctx = RuntimeContext(
            query="hello",
            limit=5,
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

        await step(ctx)

        assert len(store.calls) == 2
        for _, _, _, sf in store.calls:
            assert sf["start_date"] == "2024-01-01"
            assert sf["end_date"] == "2024-06-30"

    asyncio.run(run())


def test_search_step_invalid_date_is_ignored():
    """Invalid date strings are silently ignored (removed from filter) and search proceeds."""

    async def run():
        hit = _chunk("hit", "daily/a.md", "some text", "keyword", 5.0)
        store = FakeSearchStore(keyword_results=[hit])
        step = SearchStep(file_store=store, expand_links=False)
        ctx = RuntimeContext(
            query="hello",
            limit=5,
            start_date="abc",
        )

        resp = await step(ctx)

        assert resp.success is True
        assert len(store.calls) == 2
        for _, _, _, sf in store.calls:
            assert "start_date" not in sf

    asyncio.run(run())


def test_search_step_non_normalized_date_is_canonicalized():
    """Valid but non-canonical dates like '2024-1-5' are normalized to '2024-01-05'."""

    async def run():
        hit = _chunk("hit", "daily/a.md", "some text", "keyword", 5.0)
        store = FakeSearchStore(keyword_results=[hit])
        step = SearchStep(file_store=store, expand_links=False)
        ctx = RuntimeContext(
            query="hello",
            limit=5,
            start_date="2024-1-5",
            end_date="2024-6-1",
        )

        await step(ctx)

        for _, _, _, sf in store.calls:
            assert sf["start_date"] == "2024-01-05"
            assert sf["end_date"] == "2024-06-01"

    asyncio.run(run())


def test_search_step_date_in_search_filter_not_overridden_by_context():
    """Explicit search_filter dates take precedence over top-level context dates."""

    async def run():
        hit = _chunk("hit", "daily/a.md", "text", "keyword", 3.0)
        store = FakeSearchStore(keyword_results=[hit])
        step = SearchStep(file_store=store, expand_links=False)
        ctx = RuntimeContext(
            query="hello",
            limit=5,
            start_date="2024-01-01",
            end_date="2024-06-30",
            search_filter={"start_date": "2023-07-01", "end_date": "2023-12-31"},
        )

        await step(ctx)

        for _, _, _, sf in store.calls:
            assert sf["start_date"] == "2023-07-01"
            assert sf["end_date"] == "2023-12-31"

    asyncio.run(run())


def test_search_step_strict_date_filter_propagated_to_search_filter():
    """strict_date_filter=True is passed through to file_store via search_filter."""

    async def run():
        hit = _chunk("hit", "daily/2024-03-01/a.md", "text", "keyword", 3.0)
        store = FakeSearchStore(keyword_results=[hit])
        step = SearchStep(file_store=store, expand_links=False, strict_date_filter=True)
        ctx = RuntimeContext(
            query="hello",
            limit=5,
            start_date="2024-01-01",
            end_date="2024-06-30",
        )

        await step(ctx)

        for _, _, _, sf in store.calls:
            assert sf["strict_date_filter"] is True

    asyncio.run(run())


def test_search_step_non_strict_date_filter_not_in_search_filter():
    """strict_date_filter defaults to False and is not added to search_filter."""

    async def run():
        hit = _chunk("hit", "daily/2024-03-01/a.md", "text", "keyword", 3.0)
        store = FakeSearchStore(keyword_results=[hit])
        step = SearchStep(file_store=store, expand_links=False)
        ctx = RuntimeContext(
            query="hello",
            limit=5,
            start_date="2024-01-01",
        )

        await step(ctx)

        for _, _, _, sf in store.calls:
            assert "strict_date_filter" not in sf

    asyncio.run(run())
