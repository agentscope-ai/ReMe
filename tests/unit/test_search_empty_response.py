"""Empty search metadata explains missing results without changing answer compatibility."""

# pylint: disable=protected-access

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components import ApplicationContext
from reme.components.client.http_client import HttpClient
from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.components.service.cli_service import CliService
from reme.schema import FileChunk
from reme.steps.index import SearchStep
from reme.steps.index._source_format import ALL_RETURNED_MESSAGE, NO_RESULTS_MESSAGE


def _chunk(chunk_id="match", score=2.0):
    """Create a scored hit with an embedding that must not appear in results."""
    return FileChunk(
        id=chunk_id,
        path=f"daily/{chunk_id}.md",
        text=f"memory {chunk_id}",
        start_line=1,
        end_line=2,
        embedding=[0.1, 0.2],
        scores={"score": score},
    )


def _store(chunks=(), mode="keyword"):
    """Mock retrieval boundaries without starting indexes or model providers."""
    store = Mock(spec=BaseFileStore)
    store.vector_search = AsyncMock(return_value=list(chunks) if mode in {"vector", "hybrid"} else [])
    store.keyword_search = AsyncMock(return_value=list(chunks) if mode in {"keyword", "hybrid"} else [])
    return store


def _step(tmp_path, store, *, app_context=None, vector_weight=0.0):
    """Use an isolated application, sharing it explicitly for repeated calls."""
    return SearchStep(
        app_context=app_context if app_context is not None else ApplicationContext(workspace_dir=str(tmp_path)),
        file_store=store,
        vector_weight=vector_weight,
        expand_links=False,
    )


@pytest.mark.parametrize(
    "mode,weight,score,min_score",
    [
        ("keyword", 0.7, None, 0.0),
        ("keyword", 0.0, 0.1, 1.0),
        ("vector", 1.0, 0.1, 1.0),
        ("hybrid", 0.7, 0.1, 1.0),
    ],
    ids=["no-matches", "keyword-below-threshold", "vector-below-threshold", "hybrid-below-threshold"],
)
@pytest.mark.parametrize("tool_context_id", ["", "ctx-1"], ids=["without-context", "with-context"])
def test_search_without_eligible_results_explains_empty_metadata(
    tmp_path,
    mode,
    weight,
    score,
    min_score,
    tool_context_id,
):
    """No matches and score-filtered hits share the same empty-result behavior."""
    chunks = [] if score is None else [_chunk(score=score)]
    response = asyncio.run(
        _step(tmp_path, _store(chunks, mode), vector_weight=weight)(
            RuntimeContext(
                query="missing" if score is None else "memory",
                min_score=min_score,
                tool_context_id=tool_context_id,
            ),
        ),
    )

    assert response.success
    assert response.answer == ""
    assert response.metadata["empty_result"] == {"reason": "no_results", "message": NO_RESULTS_MESSAGE}
    assert response.metadata["results"] == []
    assert response.metadata["link_expansion"] == {}
    assert response.metadata["counts"] == {
        "vector": int(score is not None and mode in {"vector", "hybrid"}),
        "keyword": int(score is not None and mode in {"keyword", "hybrid"}),
        "returned": 0,
        "hybrid": score is not None and mode == "hybrid",
    }
    if tool_context_id:
        assert response.metadata["dedup"]["skipped_seen"] == 0
    else:
        assert "dedup" not in response.metadata


def test_search_all_seen_explains_empty_metadata_across_step_instances(tmp_path):
    """Fresh Step instances share previously returned content through the application."""

    async def run():
        app_context = ApplicationContext(workspace_dir=str(tmp_path))
        store = _store([_chunk()])
        first = await _step(tmp_path, store, app_context=app_context)(
            RuntimeContext(query="memory", tool_context_id="ctx-1"),
        )
        second = await _step(tmp_path, store, app_context=app_context)(
            RuntimeContext(query="memory", tool_context_id="ctx-1"),
        )

        assert first.metadata["counts"]["returned"] == 1
        assert "empty_result" not in first.metadata
        assert second.success
        assert second.answer == ""
        assert second.metadata["empty_result"] == {"reason": "all_returned", "message": ALL_RETURNED_MESSAGE}
        assert second.metadata["results"] == []
        assert second.metadata["link_expansion"] == {}
        assert second.metadata["counts"] == {"vector": 0, "keyword": 1, "returned": 0, "hybrid": False}
        assert second.metadata["dedup"]["seen_before"] == 1
        assert second.metadata["dedup"]["skipped_seen"] == 1

    asyncio.run(run())


@pytest.mark.parametrize("min_score", [0.0, 3.0], ids=["no-matches", "below-threshold"])
def test_search_history_does_not_turn_missing_results_into_all_returned(tmp_path, min_score):
    """Past hits do not explain a new query with no eligible candidates."""

    async def run():
        app_context = ApplicationContext(workspace_dir=str(tmp_path))
        store = _store([_chunk()])
        first = await _step(tmp_path, store, app_context=app_context)(
            RuntimeContext(query="memory", tool_context_id="ctx-1"),
        )
        assert first.metadata["counts"]["returned"] == 1
        if not min_score:
            store.keyword_search.return_value = []
        response = await _step(tmp_path, store, app_context=app_context)(
            RuntimeContext(query="another query", min_score=min_score, tool_context_id="ctx-1"),
        )

        assert response.success
        assert response.answer == ""
        assert response.metadata["empty_result"] == {"reason": "no_results", "message": NO_RESULTS_MESSAGE}
        assert response.metadata["results"] == []
        assert response.metadata["counts"]["returned"] == 0
        assert response.metadata["dedup"]["seen_before"] == 1
        assert response.metadata["dedup"]["skipped_seen"] == 0

    asyncio.run(run())


@pytest.mark.parametrize("tool_context_id", ["", "ctx-1"], ids=["without-context", "with-context"])
def test_search_hit_keeps_answer_and_structured_results(tmp_path, tool_context_id):
    """A normal hit keeps its exact rendering and excludes the embedding from metadata."""
    chunk = _chunk()
    response = asyncio.run(
        _step(tmp_path, _store([chunk]))(RuntimeContext(query="memory", tool_context_id=tool_context_id)),
    )

    assert response.success
    assert "empty_result" not in response.metadata
    assert response.answer == "========== daily/match.md:1-2 [score=2.0000] ==========\nmemory match"
    assert response.metadata["results"] == [chunk.model_dump(exclude_none=True, exclude={"embedding"})]
    assert response.metadata["link_expansion"] == {}
    assert response.metadata["counts"] == {"vector": 0, "keyword": 1, "returned": 1, "hybrid": False}


def test_search_partial_dedup_and_context_isolation_keep_hit_answers(tmp_path):
    """Unseen content still renders normally and another context can retrieve seen content."""

    async def run():
        app_context = ApplicationContext(workspace_dir=str(tmp_path))
        store = _store([_chunk()])
        await _step(tmp_path, store, app_context=app_context)(
            RuntimeContext(query="memory", tool_context_id="ctx-1"),
        )
        store.keyword_search.return_value = [_chunk(), _chunk("new")]
        response = await _step(tmp_path, store, app_context=app_context)(
            RuntimeContext(query="memory", tool_context_id="ctx-1"),
        )

        assert response.success
        assert "empty_result" not in response.metadata
        assert response.answer == "========== daily/new.md:1-2 [score=2.0000] ==========\nmemory new"
        assert [result["id"] for result in response.metadata["results"]] == ["new"]
        assert response.metadata["dedup"]["skipped_seen"] == 1

        other = await _step(tmp_path, store, app_context=app_context)(
            RuntimeContext(query="memory", tool_context_id="ctx-2"),
        )
        assert [result["id"] for result in other.metadata["results"]] == ["match", "new"]
        assert "memory match" in other.answer
        assert "memory new" in other.answer
        assert other.metadata["dedup"]["skipped_seen"] == 0
        assert "empty_result" not in other.metadata

    asyncio.run(run())


@pytest.mark.parametrize("next_query", ["memory", ""], ids=["normal-hit", "invalid-query"])
def test_search_reused_response_does_not_keep_empty_reason(tmp_path, next_query):
    """A later hit or error must not inherit an earlier empty search explanation."""

    async def run():
        store = _store()
        step = _step(tmp_path, store)
        context = RuntimeContext(query="missing")
        context.response.metadata["other_step"] = "preserved"
        first = await step(context)
        assert first.answer == ""
        assert first.metadata["empty_result"]["reason"] == "no_results"

        store.keyword_search.return_value = [_chunk()]
        context.update({"query": next_query})
        response = await step(context)

        assert "empty_result" not in response.metadata
        assert response.metadata["other_step"] == "preserved"
        if next_query:
            assert response.success
            assert "memory match" in response.answer
            assert response.metadata["counts"]["returned"] == 1
        else:
            assert not response.success
            assert response.answer == "Error: query cannot be empty"

    asyncio.run(run())


@pytest.mark.parametrize("all_returned", [False, True], ids=["no-results", "all-returned"])
@pytest.mark.parametrize("show_metadata", [False, True], ids=["default-display", "show-metadata"])
def test_empty_search_explanation_uses_existing_cli_metadata_display(tmp_path, capsys, all_returned, show_metadata):
    """Both CLI paths expose reasons only when metadata display is requested."""

    async def search():
        store = _store([_chunk()] if all_returned else [])
        step = _step(tmp_path, store)
        if all_returned:
            await step(RuntimeContext(query="memory", tool_context_id="ctx-1"))
        return await step(RuntimeContext(query="memory", tool_context_id="ctx-1"))

    response = asyncio.run(search())
    expected = {
        "reason": "all_returned" if all_returned else "no_results",
        "message": ALL_RETURNED_MESSAGE if all_returned else NO_RESULTS_MESSAGE,
    }
    assert response.answer == ""
    assert response.success
    assert response.metadata["empty_result"] == expected

    http_text = HttpClient(show_metadata=show_metadata)._format_for_display(response.model_dump_json())
    app = Mock(start=AsyncMock(), run_job=AsyncMock(return_value=response), close=AsyncMock())
    CliService(job="search", job_args={"query": "memory"}, show_metadata=show_metadata).start_service(app)
    output = capsys.readouterr()

    assert output.err == ""
    app.run_job.assert_awaited_once_with("search", query="memory")
    app.close.assert_awaited_once()
    if show_metadata:
        assert json.loads(http_text.removeprefix("\n✅ "))["empty_result"] == expected
        assert json.loads(output.out)["empty_result"] == expected
    else:
        assert http_text == "\n✅"
        assert output.out == "\n"
