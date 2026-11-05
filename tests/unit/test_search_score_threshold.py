"""Score-scale diagnostics preserve existing search threshold behavior."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from reme.components.file_store import BaseFileStore
from reme.components.runtime_context import RuntimeContext
from reme.schema import FileChunk
from reme.steps.index import SearchStep


@pytest.mark.parametrize(
    "mode,weight,min_score,expected_ids,warn",
    [
        ("hybrid", 0.7, 0.05, [], True),
        ("hybrid", 0.2, 0.05, [], True),
        ("hybrid", 0.7, 1 / 61, ["match"], False),
        ("hybrid", 0.7, 0.01, ["match"], False),
        ("hybrid", 0.7, 0, ["match"], False),
        ("vector", 0.7, 0.05, ["match"], False),
        ("keyword", 0.7, 1, ["match"], False),
        ("vector", 1, 0.05, ["match"], False),
        ("keyword", 0, 1, ["match"], False),
        ("empty", 0.7, 0.05, [], False),
    ],
)
def test_search_score_threshold_warning(mode, weight, min_score, expected_ids, warn):
    """Warn only above the hybrid bound, preserving native scores and exact-bound hits."""
    vector = FileChunk(
        id="match",
        path="daily/match.md",
        text="matching memory",
        start_line=1,
        end_line=1,
        scores={"vector": 0.9, "score": 0.9},
    )
    keyword = vector.model_copy(update={"scores": {"keyword": 8.0, "score": 8.0}})
    store = Mock(spec=BaseFileStore)
    store.vector_search = AsyncMock(return_value=[vector] if mode in {"hybrid", "vector"} else [])
    store.keyword_search = AsyncMock(return_value=[keyword] if mode in {"hybrid", "keyword"} else [])
    step = SearchStep(file_store=store, vector_weight=weight, expand_links=False)
    step.logger = Mock()

    response = asyncio.run(step(RuntimeContext(query="memory", min_score=min_score)))

    assert response.success
    assert [result["id"] for result in response.metadata["results"]] == expected_ids
    assert step.logger.warning.call_count == int(warn)
    if warn:
        message = step.logger.warning.call_args.args[0]
        assert "min_score=0.05 exceeds the hybrid RRF maximum" in message
        assert "all hybrid results will be filtered out" in message
        assert "min_score=0" in message
    elif expected_ids:
        expected_score = {"hybrid": 1 / 61, "vector": 0.9, "keyword": 8.0}[mode]
        assert response.metadata["results"][0]["scores"]["score"] == pytest.approx(expected_score)
