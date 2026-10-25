"""Fault injection across file commits, two indexes and segment checkpoints."""

# pylint: disable=missing-function-docstring,protected-access,redefined-outer-name

import json

import pytest

from reme_procedure_memory import auto_memory, store as store_module, trajectory
from reme_procedure_memory.checkpoint import load_json


@pytest.mark.asyncio
async def test_partial_index_failure_replays_without_recreating(runtime, experience, monkeypatch):
    _, store, scope = runtime
    second = store.views["when_to_use_content"]
    upsert = second.upsert

    async def fail(_files):
        raise RuntimeError("second index failed")

    monkeypatch.setattr(second, "upsert", fail)
    with pytest.raises(RuntimeError, match="File committed, indexes pending"):
        await store.apply_mutation("create", experience, scope)
    assert len(list(store.procedure_root.glob("*.json"))) == 1
    assert not store.views["when_to_use"].file_chunks
    assert not second.file_chunks
    partial = await store.search("repair", "when_to_use")
    assert not partial["index_complete"]
    assert partial["results"] == []
    assert partial["pending_indexes"][0]["path"] == "procedures/repair-search-index.json"
    monkeypatch.setattr(second, "upsert", upsert)
    recovered = await store.apply_mutation("create", experience, scope)
    assert recovered["status"] == "indexed"
    assert len(list(store.procedure_root.glob("*.json"))) == 1
    assert (await store.search("repair", "when_to_use_content"))["results"]


@pytest.mark.asyncio
async def test_prepared_operation_survives_write_crash(runtime, experience, monkeypatch):
    _, store, scope = runtime
    original = store_module.atomic_json

    def crash(path, value):
        if path.parent == store.procedure_root:
            raise RuntimeError("write interrupted")
        original(path, value)

    monkeypatch.setattr(store_module, "atomic_json", crash)
    with pytest.raises(RuntimeError, match="write interrupted"):
        await store.apply_mutation("create", experience, scope)
    assert not list(store.procedure_root.glob("*.json"))
    records = store.operation_records("session/trajectories/input.log", 0)
    assert records[0]["status"] == "prepared"
    monkeypatch.setattr(store_module, "atomic_json", original)
    await store.reconcile()
    assert (await store.apply_mutation("create", experience, scope))["status"] == "indexed"
    assert len(list(store.procedure_root.glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_restart_rebuilds_from_json(app_factory, experience, tmp_path):
    app, store = await app_factory()
    source = tmp_path / "a.txt"
    source.write_text("evidence")
    record = store.import_trajectory(str(source), "a.txt", "restart")
    scope = store.open_scope(record["source"], 0)
    await store.apply_mutation("create", experience, scope)
    await app.close()
    # Deliberately destroy all derived file-store data after a clean shutdown.
    for path in (store.workspace_metadata_path / "file_store").glob("*"):
        path.unlink()
    _, reopened = await app_factory(workspace=store.workspace_path)
    for view in ("when_to_use", "when_to_use_content"):
        results = (await reopened.search("repair", view))["results"]
        assert results[0]["path"] == "procedures/repair-search-index.json"
        assert results[0]["experience"]["sources"] == [record["source"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("original", ["unchanged", "deleted", "moved"])
async def test_resume_with_saved_tool_context(app_factory, experience, tmp_path, original):
    app, store = await app_factory()
    source = tmp_path / "a.txt"
    source.write_text("repair")
    calls = 0

    async def reply(inputs, options):
        nonlocal calls
        calls += 1
        data = json.loads(inputs)
        if calls == 1:
            result = await app.context.jobs["procedure_create"](**experience, **options["injected_job_kwargs"])
            assert result.success
            raise RuntimeError("model crashed after successful tool call")
        assert len(data["completed_operations"]) == 1
        assert data["completed_operations"][0]["operation_key"] == experience["operation_key"]
        read = await app.context.jobs["procedure_read"](path=data["completed_operations"][0]["path"])
        assert read.success
        return {"result": '{"summary":"recovered","complete":true}'}

    app.context.metadata["fake_reply"] = reply
    job = app.context.jobs["procedure_auto_memory"]
    first = await job(file=str(source), filename="a.txt", request_id="resume")
    assert not first.success
    if original != "unchanged":
        if original == "deleted":
            source.unlink()
        else:
            source.rename(tmp_path / "moved.txt")
        workspace = store.workspace_path
        await app.close()
        app, store = await app_factory(workspace=workspace)
        app.context.metadata["fake_reply"] = reply
        job = app.context.jobs["procedure_auto_memory"]
    second = await job(file=str(source), filename="a.txt", request_id="resume")
    assert second.success, second.answer
    assert len(list(store.procedure_root.glob("*.json"))) == 1
    assert len(list(store.trajectory_root.glob("*"))) == 1
    assert calls == 2


@pytest.mark.asyncio
async def test_reply_saved_before_checkpoint_advances(app_factory, tmp_path, monkeypatch):
    app, store = await app_factory()
    source = tmp_path / "a.txt"
    source.write_text("evidence")
    calls = 0

    async def reply(_inputs, _options):
        nonlocal calls
        calls += 1
        return {"result": '{"summary":"saved reply","complete":true}'}

    app.context.metadata["fake_reply"] = reply
    original = auto_memory.atomic_json

    def crash(path, value):
        if value.get("next_segment") == 1:
            raise RuntimeError("crash before advancing checkpoint")
        original(path, value)

    monkeypatch.setattr(auto_memory, "atomic_json", crash)
    job = app.context.jobs["procedure_auto_memory"]
    assert not (await job(file=str(source), filename="a.txt", request_id="checkpoint")).success
    checkpoint = load_json(store.checkpoint_path("session/trajectories/a.txt"))
    assert checkpoint["next_segment"] == 0
    assert checkpoint["active"]["reply"]
    monkeypatch.setattr(auto_memory, "atomic_json", original)
    assert (await job(file=str(source), filename="a.txt", request_id="checkpoint")).success
    assert calls == 1


@pytest.mark.asyncio
async def test_incomplete_is_resumable_and_config_change_rejected(app_factory, tmp_path):
    app, store = await app_factory()
    source = tmp_path / "a.txt"
    source.write_text("evidence")
    calls = 0

    async def reply(inputs, _options):
        nonlocal calls
        calls += 1
        if calls > 1:
            assert json.loads(inputs)["unfinished_summary"] == "unfinished"
        return {"result": json.dumps({"summary": "unfinished", "complete": calls > 1})}

    app.context.metadata["fake_reply"] = reply
    job = app.context.jobs["procedure_auto_memory"]
    first = await job(file=str(source), filename="a.txt", request_id="incomplete")
    assert not first.success
    assert first.metadata["next_segment"] == 0
    mismatch = await job(file=str(source), filename="a.txt", request_id="incomplete", memory_hint="different")
    assert not mismatch.success
    assert "configuration changed" in mismatch.metadata["error"]
    assert calls == 1
    second = await job(file=str(source), filename="a.txt", request_id="incomplete")
    assert second.success
    assert not store.scopes


@pytest.mark.asyncio
async def test_failed_import_keeps_allocated_name(runtime, tmp_path, monkeypatch):
    _, store, _ = runtime
    source = tmp_path / "a.txt"
    source.write_text("evidence")
    link = trajectory.os.link

    def fail(*_args):
        raise RuntimeError("copy interrupted")

    monkeypatch.setattr(trajectory.os, "link", fail)
    with pytest.raises(RuntimeError):
        store.import_trajectory(str(source), "a.txt", "copy")
    assert not (store.trajectory_root / "a.txt").exists()
    assert not list(store.trajectory_root.glob(".trajectory-*"))
    monkeypatch.setattr(trajectory.os, "link", link)
    record = store.import_trajectory(str(source), "a.txt", "copy")
    assert record["source"] == "session/trajectories/a.txt"


@pytest.mark.asyncio
async def test_unsupported_import_reports_actual_path(app_factory, tmp_path):
    app, store = await app_factory()
    source = tmp_path / "a.pdf"
    source.write_bytes(b"%PDF-unsupported")
    result = await app.context.jobs["procedure_auto_memory"](
        file=str(source),
        filename="a.pdf",
        request_id="unsupported",
    )
    assert not result.success
    assert result.metadata["source"] == "session/trajectories/a.pdf"
    assert "Unsupported" in result.metadata["error"]
    assert (store.trajectory_root / "a.pdf").read_bytes() == source.read_bytes()


@pytest.mark.asyncio
async def test_pending_import_name_is_reserved(runtime, tmp_path, monkeypatch):
    _, store, _ = runtime
    source = tmp_path / "a.txt"
    source.write_text("evidence")
    link = trajectory.os.link

    def fail(*_args):
        raise RuntimeError("copy interrupted")

    monkeypatch.setattr(trajectory.os, "link", fail)
    with pytest.raises(RuntimeError):
        store.import_trajectory(str(source), "a.txt", "pending")
    monkeypatch.setattr(trajectory.os, "link", link)
    second = store.import_trajectory(str(source), "a.txt", "independent")
    assert second["source"] == "session/trajectories/a-2.txt"
    first = store.import_trajectory(str(source), "a.txt", "pending")
    assert first["source"] == "session/trajectories/a.txt"


@pytest.mark.asyncio
async def test_silent_core_embedding_failure_is_reported(runtime, experience, monkeypatch):
    _, store, scope = runtime
    embedding = store.views["when_to_use"].embedding_store
    original = embedding.get_embeddings

    async def missing(input_text, **_kwargs):
        return [None for _ in input_text]

    monkeypatch.setattr(embedding, "get_embeddings", missing)
    with pytest.raises(RuntimeError, match="File committed, indexes pending"):
        await store.apply_mutation("create", experience, scope)
    assert len(list(store.procedure_root.glob("*.json"))) == 1
    monkeypatch.setattr(embedding, "get_embeddings", original)
    await store.reconcile()
    assert (await store.search("repair", "when_to_use"))["results"]


@pytest.mark.asyncio
async def test_context_capacity_can_be_increased_to_resume(app_factory, tmp_path):
    app, store = await app_factory(token_limits={"context": 100, "reserve": 10})
    source = tmp_path / "a.txt"
    source.write_text("evidence")

    async def reply(_inputs, _options):
        return {"result": '{"summary":"done","complete":true}'}

    app.context.metadata["fake_reply"] = reply
    job = app.context.jobs["procedure_auto_memory"]
    first = await job(file=str(source), filename="a.txt", request_id="context-budget")
    assert not first.success
    assert "context budget" in first.metadata["error"]
    store.budget.limits["context"] = 32768
    assert (await job(file=str(source), filename="a.txt", request_id="context-budget")).success
    assert len(list(store.trajectory_root.glob("*"))) == 1


@pytest.mark.asyncio
async def test_rebuild_clear_failure_cannot_serve_partial_results(runtime, experience, monkeypatch):
    app, store, scope = runtime
    await store.apply_mutation("create", experience, scope)
    second = store.views["when_to_use_content"]
    original = second.clear

    async def fail():
        raise RuntimeError("second clear interrupted")

    monkeypatch.setattr(second, "clear", fail)
    result = await app.context.jobs["procedure_reindex"]()
    assert not result.success
    assert store.rebuild_pending
    with pytest.raises(RuntimeError, match="second clear interrupted"):
        await store.search("repair", "when_to_use")
    monkeypatch.setattr(second, "clear", original)
    assert (await store.search("repair", "when_to_use"))["results"]
    assert not store.rebuild_pending
    assert all(len(index.file_chunks) == 1 for index in store.views.values())


@pytest.fixture
def reject_payload(monkeypatch):
    """Permanently reject one payload across component restarts, not the provider itself."""

    def install(embedding):
        backend = type(embedding)
        original = backend.get_embeddings

        async def selective(self, texts, **kwargs):
            if any("unembeddable-payload" in text for text in texts):
                self.is_healthy = False
                return [None for _ in texts]
            self.is_healthy = True
            return await original(self, texts, **kwargs)

        monkeypatch.setattr(backend, "get_embeddings", selective)

    return install


@pytest.mark.asyncio
async def test_permanent_failure_can_be_corrected_without_blocking_other_files(runtime, experience, reject_payload):
    app, store, scope = runtime
    reject_payload(store.views["when_to_use"].embedding_store)
    healthy = await store.apply_mutation("create", experience, scope)
    bad_arguments = {
        **experience,
        "filename": "needs-correction.json",
        "operation_key": "bad-create",
        "content": "unembeddable-payload",
    }
    failed = await app.context.jobs["procedure_create"](**bad_arguments, _procedure_scope=scope)
    assert not failed.success
    assert failed.metadata["status"] == "index_pending"
    assert failed.metadata["version"]
    assert "new operation_key" in json.loads(failed.answer)["repair"]
    path = failed.metadata["path"]
    saved = await store.read_experience(path)
    assert saved["experience"]["content"] == "unembeddable-payload"
    partial = await store.search("repair", "when_to_use")
    assert not partial["index_complete"]
    assert [hit["path"] for hit in partial["results"]] == [healthy["path"]]
    assert partial["pending_indexes"][0]["path"] == path
    for index in store.views.values():
        assert path not in {chunk.path for chunk in index.file_chunks.values()}
    await store.apply_mutation(
        "create",
        {
            **experience,
            "filename": "other-valid-file.json",
            "operation_key": "unrelated",
        },
        scope,
    )
    correction = {
        "path": path,
        "expected_version": saved["version"],
        "changes": {"content": "corrected recommendation"},
        "operation_key": "correct-payload",
    }
    with pytest.raises(ValueError, match="Version conflict"):
        await store.apply_mutation("update", {**correction, "expected_version": "outdated"}, scope)
    fixed = await app.context.jobs["procedure_update"](**correction, _procedure_scope=scope)
    assert fixed.success, fixed.answer
    assert fixed.metadata["path"] == path
    assert (await store.read_experience(path))["experience"]["sources"] == saved["experience"]["sources"]
    assert await store.apply_mutation("update", correction, scope) == fixed.metadata
    repeated_old = await store.apply_mutation("create", bad_arguments, scope)
    assert repeated_old["status"] == "superseded"
    assert (await store.read_experience(path))["experience"]["content"] == "corrected recommendation"
    complete = await store.search("repair", "when_to_use_content")
    assert complete["index_complete"]
    assert complete["pending_indexes"] == []
    for index in store.views.values():
        assert (
            next(chunk for chunk in index.file_chunks.values() if chunk.path == path).metadata["version"]
            == fixed.metadata["version"]
        )


@pytest.mark.asyncio
async def test_pending_payload_survives_restart_and_can_still_be_repaired(
    runtime,
    experience,
    app_factory,
    reject_payload,
):
    app, store, scope = runtime
    reject_payload(store.views["when_to_use"].embedding_store)
    arguments = {**experience, "content": "unembeddable-payload"}
    with pytest.raises(RuntimeError, match="indexes pending"):
        await store.apply_mutation("create", arguments, scope)
    await app.close()
    reopened_app, reopened = await app_factory(workspace=store.workspace_path)
    assert reopened.is_started
    pending = await reopened.search("repair", "when_to_use_content")
    assert not pending["index_complete"]
    assert pending["results"] == []
    saved = await reopened.read_experience("procedures/repair-search-index.json")
    new_scope = reopened.open_scope(saved["experience"]["sources"][0], 0)
    corrected = await reopened.apply_mutation(
        "update",
        {
            "path": saved["path"],
            "expected_version": saved["version"],
            "changes": {"content": "valid correction"},
            "operation_key": "restart-correction",
        },
        new_scope,
    )
    await reopened_app.close()
    _, final = await app_factory(workspace=store.workspace_path)
    assert (await final.read_experience(saved["path"]))["version"] == corrected["version"]
    assert (await final.search("repair", "when_to_use_content"))["index_complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["supersede", "publish", "indexed"])
async def test_repair_intent_recovers_without_replaying_old_payload(
    runtime,
    experience,
    reject_payload,
    monkeypatch,
    stage,
):
    _, store, scope = runtime
    reject_payload(store.views["when_to_use"].embedding_store)
    with pytest.raises(RuntimeError, match="indexes pending"):
        await store.apply_mutation("create", {**experience, "content": "unembeddable-payload"}, scope)
    saved = await store.read_experience("procedures/repair-search-index.json")
    correction = {
        "path": saved["path"],
        "expected_version": saved["version"],
        "changes": {"content": "valid replacement"},
        "operation_key": "repair",
    }
    original = store_module.atomic_json

    def crash(path, value):
        if stage == "supersede" and value.get("status") == "superseded":
            raise RuntimeError("repair interrupted")
        if stage == "publish" and path.parent == store.procedure_root:
            raise RuntimeError("repair interrupted")
        if stage == "indexed" and value.get("operation_key") == "repair" and value.get("status") == "indexed":
            raise RuntimeError("repair interrupted")
        original(path, value)

    monkeypatch.setattr(store_module, "atomic_json", crash)
    with pytest.raises(RuntimeError, match="repair interrupted"):
        await store.apply_mutation("update", correction, scope)
    monkeypatch.setattr(store_module, "atomic_json", original)
    await store.reconcile()
    assert (await store.apply_mutation("update", correction, scope))["status"] == "indexed"
    assert (await store.read_experience(saved["path"]))["experience"]["content"] == "valid replacement"
    statuses = {
        record["operation_key"]: record["status"]
        for record in store.operation_records("session/trajectories/input.log", 0)
    }
    assert statuses == {experience["operation_key"]: "superseded", "repair": "indexed"}
    assert (await store.search("repair", "when_to_use_content"))["index_complete"]


@pytest.mark.asyncio
async def test_pending_segment_runs_agent_again_for_repair(app_factory, tmp_path, experience, reject_payload):
    app, store = await app_factory()
    reject_payload(store.views["when_to_use"].embedding_store)
    source = tmp_path / "repair.log"
    source.write_text("evidence")
    calls = 0

    async def reply(inputs, options):
        nonlocal calls
        calls += 1
        if calls == 1:
            failed = await app.context.jobs["procedure_create"](
                **{**experience, "content": "unembeddable-payload"},
                **options["injected_job_kwargs"],
            )
            assert not failed.success
        else:
            data = json.loads(inputs)
            assert data["completed_operations"][0]["status"] == "committed"
            assert "index_error" in data["completed_operations"][0]
            current = await app.context.jobs["procedure_read"](path=data["completed_operations"][0]["path"])
            fixed = await app.context.jobs["procedure_update"](
                path=current.metadata["path"],
                expected_version=current.metadata["version"],
                changes={"content": "fixed from evidence"},
                operation_key="repair",
                **options["injected_job_kwargs"],
            )
            assert fixed.success, fixed.answer
        return {"result": '{"summary":"processed evidence","complete":true}'}

    app.context.metadata["fake_reply"] = reply
    job = app.context.jobs["procedure_auto_memory"]
    first = await job(file=str(source), filename="repair.log", request_id="same-request")
    assert not first.success
    checkpoint = load_json(store.checkpoint_path("session/trajectories/repair.log"))
    assert checkpoint["next_segment"] == 0
    assert "reply" not in checkpoint["active"]
    second = await job(file=str(source), filename="repair.log", request_id="same-request")
    assert second.success, second.answer
    assert calls == 2

    # Another trajectory can complete without fixing unrelated pending writes.
    with pytest.raises(RuntimeError, match="indexes pending"):
        await store.apply_mutation(
            "create",
            {
                **experience,
                "filename": "unrelated-pending.json",
                "operation_key": "unrelated",
                "content": "unembeddable-payload",
            },
            store.open_scope("session/trajectories/repair.log", 0),
        )

    async def skip(_inputs, _options):
        return {"result": '{"summary":"nothing reusable","complete":true}'}

    app.context.metadata["fake_reply"] = skip
    unrelated = await job(file=str(source), filename="different.log", request_id="different")
    assert unrelated.success, unrelated.answer


@pytest.mark.asyncio
async def test_real_embedding_cache_does_not_block_correction(runtime, experience, monkeypatch):
    from reme.components.embedding_store.local_embedding_store import LocalEmbeddingStore

    _, store, scope = runtime

    class Provider:
        """Return real vectors or an explicit per-input rejection without network calls."""

        dimensions = 3
        vector_space_id = "repair-review"

        async def __call__(self, texts, **_kwargs):
            return [None if "unembeddable-payload" in text else [1.0, 0.0, 0.0] for text in texts]

    real = LocalEmbeddingStore(app_context=store.app_context, max_retries=1)
    real.as_embedding = Provider()
    for view in store.views.values():
        monkeypatch.setattr(view, "embedding_store", real)
    # Cache the healthy query before a payload failure flips provider health.
    await store.apply_mutation("create", experience, scope)
    await store.search("repair", "when_to_use")
    with pytest.raises(RuntimeError, match="indexes pending"):
        await store.apply_mutation(
            "create",
            {
                **experience,
                "filename": "cached-repair.json",
                "operation_key": "bad",
                "content": "unembeddable-payload",
            },
            scope,
        )
    assert not real.is_healthy
    assert (await store.search("repair", "when_to_use"))["results"]
    current = await store.read_experience("procedures/cached-repair.json")
    fixed = await store.apply_mutation(
        "update",
        {
            "path": current["path"],
            "expected_version": current["version"],
            "operation_key": "repair-real",
            "changes": {"content": "valid replacement"},
        },
        scope,
    )
    assert fixed["status"] == "indexed"
    assert (await store.search("repair", "when_to_use_content"))["index_complete"]


@pytest.mark.asyncio
async def test_failed_update_hides_previous_projection_and_allows_another_correction(
    runtime,
    experience,
    reject_payload,
):
    _, store, scope = runtime
    reject_payload(store.views["when_to_use"].embedding_store)
    original = await store.apply_mutation("create", experience, scope)
    with pytest.raises(RuntimeError, match="indexes pending"):
        await store.apply_mutation(
            "update",
            {
                "path": original["path"],
                "expected_version": original["version"],
                "operation_key": "bad-update",
                "changes": {"content": "unembeddable-payload"},
            },
            scope,
        )
    pending = await store.search("repair", "when_to_use_content")
    assert pending["results"] == []
    assert not pending["index_complete"]
    current = await store.read_experience(original["path"])
    # A correction that is still rejected must replace, rather than revive, its predecessor.
    with pytest.raises(RuntimeError, match="indexes pending"):
        await store.apply_mutation(
            "update",
            {
                "path": current["path"],
                "expected_version": current["version"],
                "operation_key": "bad-correction",
                "changes": {"content": "still-unembeddable-payload"},
            },
            scope,
        )
    current = await store.read_experience(original["path"])
    await store.apply_mutation(
        "update",
        {
            "path": current["path"],
            "expected_version": current["version"],
            "operation_key": "good-correction",
            "changes": {"content": "valid procedure"},
        },
        scope,
    )
    await store.reconcile()
    result = await store.search("repair", "when_to_use_content")
    assert result["index_complete"]
    assert result["results"][0]["experience"]["content"] == "valid procedure"
