"""Experience writes, source merging, version checks and exact index projections."""

# pylint: disable=missing-function-docstring

import asyncio
import json

import pytest


@pytest.mark.asyncio
async def test_create_update_and_projection(runtime, experience, tmp_path):
    app, store, scope = runtime
    created = await store.apply_mutation("create", experience, scope)
    relative = created["path"]
    assert relative == "procedures/repair-search-index.json"
    original = await store.read_experience(relative)
    assert original["experience"]["sources"] == ["session/trajectories/input.log"]
    assert not {"id", "experience-id", "path"} & original["experience"].keys()
    for view, index in store.views.items():
        chunks = list(index.file_chunks.values())
        assert len(chunks) == 1
        assert chunks[0].path == relative
        expected = experience["when-to-use"] + ("\n\n" + experience["content"] if view == "when_to_use_content" else "")
        assert chunks[0].text == expected
        assert chunks[0].embedding is not None
        assert experience["description"] not in chunks[0].text
    source = tmp_path / "second.md"
    source.write_text("new evidence")
    imported = store.import_trajectory(str(source), "second.md", "second")
    second_scope = store.open_scope(imported["source"], 0)
    updated = await store.apply_mutation(
        "update",
        {
            "path": relative,
            "expected_version": original["version"],
            "operation_key": "revise",
            "changes": {
                "description": "Updated",
                "when-to-use": "After failed repair",
                "content": "Inspect logs before rebuilding",
            },
        },
        second_scope,
    )
    assert updated["path"] == relative
    assert updated["version"] != original["version"]
    current = await store.read_experience(relative)
    assert current["experience"]["sources"] == ["session/trajectories/input.log", "session/trajectories/second.md"]
    for index in store.views.values():
        assert len(index.file_chunks) == 1
        assert next(iter(index.file_chunks.values())).metadata["version"] == updated["version"]
    hit = await app.context.jobs["procedure_search"](query="failed repair", index="when_to_use")
    assert hit.success
    assert hit.metadata["results"][0]["experience"] == current["experience"]
    assert not list(store.trajectory_root.glob("source.json"))


@pytest.mark.asyncio
async def test_idempotency_collision_and_optimistic_update(runtime, experience):
    _, store, scope = runtime
    first = await store.apply_mutation("create", experience, scope)
    assert await store.apply_mutation("create", experience, scope) == first
    with pytest.raises(ValueError, match="different arguments"):
        await store.apply_mutation("create", {**experience, "content": "different"}, scope)
    with pytest.raises(ValueError, match="Experience exists"):
        await store.apply_mutation("create", {**experience, "operation_key": "other"}, scope)
    update = {"path": first["path"], "expected_version": first["version"], "changes": {"content": "Repaired"}}
    results = await asyncio.gather(
        *[
            store.apply_mutation(
                "update",
                {**update, "operation_key": f"update-{i}", "changes": {"content": f"Repaired {i}"}},
                scope,
            )
            for i in range(2)
        ],
        return_exceptions=True,
    )
    assert sum(isinstance(result, ValueError) for result in results) == 1
    current = await store.read_experience(first["path"])
    assert current["experience"]["sources"] == ["session/trajectories/input.log"]


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["../bad.json", "Bad Name.json", "id_1.json", "a.txt", "a/b.json"])
async def test_semantic_filename(runtime, experience, filename):
    _, store, scope = runtime
    with pytest.raises(ValueError):
        await store.apply_mutation("create", {**experience, "filename": filename}, scope)


@pytest.mark.asyncio
@pytest.mark.parametrize("field,ceiling", [("description", 140), ("when-to-use", 563), ("content", 2252)])
async def test_field_slack_on_create_and_update(runtime, experience, field, ceiling):
    _, store, scope = runtime
    with pytest.raises(ValueError, match="Rewrite it more concisely"):
        await store.apply_mutation("create", {**experience, field: "x" * (ceiling + 1)}, scope)
    assert not list(store.procedure_root.glob("*.json"))
    created = await store.apply_mutation("create", {**experience, field: "x" * ceiling}, scope)
    update = {
        "path": created["path"],
        "expected_version": created["version"],
        "operation_key": "update-slack",
        "changes": {field: "y" * (ceiling + 1)},
    }
    with pytest.raises(ValueError, match="Rewrite it more concisely"):
        await store.apply_mutation("update", update, scope)
    assert (await store.read_experience(created["path"]))["version"] == created["version"]
    await store.apply_mutation("update", {**update, "changes": {field: "y" * ceiling}}, scope)
    assert (await store.read_experience(created["path"]))["experience"][field] == "y" * ceiling


@pytest.mark.asyncio
async def test_field_limits_and_sources_cannot_be_overwritten(runtime, experience):
    _, store, scope = runtime
    store.budget.limits["content"] = 10
    with pytest.raises(ValueError, match="maximum is 10"):
        await store.apply_mutation("create", experience, scope)
    assert not list(store.procedure_root.glob("*.json"))
    store.budget.limits["content"] = 100
    created = await store.apply_mutation("create", experience, scope)
    with pytest.raises(ValueError, match="changes may contain only"):
        await store.apply_mutation(
            "update",
            {
                "path": created["path"],
                "expected_version": created["version"],
                "operation_key": "source-spoof",
                "changes": {"sources": ["fake"]},
            },
            scope,
        )
    assert json.loads((store.procedure_root / experience["filename"]).read_text())["content"] == experience["content"]


@pytest.mark.asyncio
async def test_invalid_reindex_reports_without_modifying_files(runtime, experience):
    app, store, scope = runtime
    await store.apply_mutation("create", experience, scope)
    corrupt = store.procedure_root / "invalid.json"
    corrupt.write_text('{"incomplete":true}')
    for index in store.views.values():
        await index.clear()
    result = await app.context.jobs["procedure_reindex"]()
    assert not result.success
    assert result.metadata["indexed"] == 1
    assert result.metadata["errors"][0]["path"] == "procedures/invalid.json"
    assert corrupt.read_text() == '{"incomplete":true}'
    assert all(len(index.file_chunks) == 1 for index in store.views.values())
