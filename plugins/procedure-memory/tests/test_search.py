"""Selectable hybrid search, complete current documents and restricted reads."""

# pylint: disable=missing-function-docstring

import json
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_search_choices_repeat_and_full_answers(runtime, experience):
    app, store, scope = runtime
    await store.apply_mutation("create", {**experience, "content": "zebraspecial"}, scope)
    assert (await store.search("zebraspecial", "when_to_use", vector_weight=0))["results"] == []
    first = await store.search("zebraspecial", "when_to_use_content", vector_weight=0)
    assert first["results"][0]["experience"]["content"] == "zebraspecial"
    assert await store.search("zebraspecial", "when_to_use_content", vector_weight=0) == first
    result = await app.context.jobs["procedure_search"](query="configuration", index="when_to_use")
    assert json.loads(result.answer)["results"][0]["experience"]["content"] == "zebraspecial"
    with pytest.raises(ValueError, match="index must be"):
        await store.search("hello", "invalid")


@pytest.mark.asyncio
async def test_weighted_rrf(runtime, experience, monkeypatch):
    _, store, scope = runtime
    await store.apply_mutation("create", experience, scope)
    await store.apply_mutation(
        "create",
        {**experience, "filename": "another-repair.json", "operation_key": "two"},
        scope,
    )
    index = store.views["when_to_use"]
    chunks = list(index.file_chunks.values())

    async def vector(*_args):
        return chunks

    async def keyword(*_args):
        return list(reversed(chunks))

    monkeypatch.setattr(index, "vector_search", vector)
    monkeypatch.setattr(index, "keyword_search", keyword)
    results = (await store.search("repair", "when_to_use"))["results"]
    assert results[0]["path"] == chunks[0].path
    assert results[0]["score"] == pytest.approx(0.7 / 61 + 0.3 / 62)


@pytest.mark.asyncio
@pytest.mark.parametrize("vector_weight", [0, 1])
async def test_disabled_search_channel_is_not_called(runtime, experience, monkeypatch, vector_weight):
    _, store, scope = runtime
    await store.apply_mutation("create", experience, scope)
    index = store.views["when_to_use"]
    chunks = list(index.file_chunks.values())
    enabled = AsyncMock(return_value=chunks)
    disabled = AsyncMock(side_effect=RuntimeError("disabled channel unavailable"))
    monkeypatch.setattr(index, "vector_search", enabled if vector_weight == 1 else disabled)
    monkeypatch.setattr(index, "keyword_search", enabled if vector_weight == 0 else disabled)

    results = (await store.search("repair", "when_to_use", vector_weight=vector_weight))["results"]

    enabled.assert_awaited_once()
    disabled.assert_not_awaited()
    assert results[0]["path"] == chunks[0].path
    assert results[0]["score"] == pytest.approx(1 / 61)


@pytest.mark.asyncio
async def test_bounded_source_read_and_path_permissions(runtime):
    app, _, _ = runtime
    job = app.context.jobs["procedure_read"]
    first = await job(path="session/trajectories/input.log", kind="trajectory", offset=0, length=5)
    assert first.success
    assert first.metadata["text"] == "repai"
    assert first.metadata["more"]
    second = await job(path="session/trajectories/input.log", kind="trajectory", offset=5, length=500)
    assert first.metadata["text"] + second.metadata["text"] == "repair an index\n"
    assert not second.metadata["more"]
    for path in ("../outside.txt", "/etc/hosts", "metadata/secret", "session/dialog/private.jsonl"):
        assert not (await job(path=path, kind="trajectory")).success
