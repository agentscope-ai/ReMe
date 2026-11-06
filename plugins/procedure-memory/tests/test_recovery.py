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
async def test_restart_recovers_acknowledged_write_without_cache_flush(runtime, experience, app_factory):
    _, store, scope = runtime
    created = await store.apply_mutation("create", experience, scope)
    # Reopen before the first application's close, as after losing its in-memory caches.
    _, reopened = await app_factory(workspace=store.workspace_path)
    for view in reopened.views:
        results = (await reopened.search("repair", view))["results"]
        assert [result["path"] for result in results] == [created["path"]]
        assert results[0]["version"] == created["version"]
    new_scope = reopened.open_scope("session/trajectories/input.log", 0)
    assert await reopened.apply_mutation("create", experience, new_scope) == created


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
@pytest.mark.parametrize("operation_count", [0, 20])
async def test_reply_saved_before_checkpoint_advances(app_factory, tmp_path, monkeypatch, operation_count):
    app, store = await app_factory(token_limits={"context": 8000, "reserve": 100})
    context_checks = []
    check_context = auto_memory.ProcedureAutoMemoryStep._check_context

    def track_context(step, message, tools):
        context_checks.append((step, message, tools))
        check_context(step, message, tools)

    monkeypatch.setattr(auto_memory.ProcedureAutoMemoryStep, "_check_context", track_context)
    source = tmp_path / "a.txt"
    source.write_text("evidence")
    calls = 0

    async def reply(_inputs, options):
        nonlocal calls
        calls += 1
        for number in range(operation_count):
            result = await app.context.jobs["procedure_create"](
                filename=f"experience-{number}.json",
                description="Repair search",
                **{"when-to-use": "After an index configuration change"},
                content="Rebuild the index before searching",
                operation_key=f"create-{number}",
                **options["injected_job_kwargs"],
            )
            assert result.success
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
    assert len(store.operation_records("session/trajectories/a.txt", 0)) == operation_count
    if operation_count:
        step, message, tools = context_checks[0]
        recovered_input = json.loads(message)
        recovered_input["completed_operations"] = store.operation_records("session/trajectories/a.txt", 0)
        with pytest.raises(ValueError, match="context budget"):
            check_context(step, auto_memory.json_text(recovered_input), tools)
    monkeypatch.setattr(auto_memory, "atomic_json", original)
    assert (await job(file=str(source), filename="a.txt", request_id="checkpoint")).success
    assert calls == 1
    assert len(context_checks) == 1
    checkpoint = load_json(store.checkpoint_path("session/trajectories/a.txt"))
    assert checkpoint["complete"]
    assert checkpoint["next_segment"] == 1
    assert checkpoint["summary"] == "saved reply"


@pytest.mark.asyncio
async def test_segmentation_version_change_rejects_resume(app_factory, tmp_path, monkeypatch):
    app, store = await app_factory()
    source = tmp_path / "segmentation.log"
    source.write_text("evidence")
    calls = 0

    async def reply(_inputs, _options):
        nonlocal calls
        calls += 1
        return {"result": '{"summary":"unfinished","complete":false}'}

    app.context.metadata["fake_reply"] = reply
    job = app.context.jobs["procedure_auto_memory"]
    arguments = {"file": str(source), "filename": "segmentation.log", "request_id": "segmentation"}
    with monkeypatch.context() as patch:
        patch.setattr(auto_memory, "SEGMENTATION_VERSION", auto_memory.SEGMENTATION_VERSION - 1)
        assert not (await job(**arguments)).success
    checkpoint_path = store.checkpoint_path("session/trajectories/segmentation.log")
    checkpoint = checkpoint_path.read_bytes()
    result = await job(**arguments)
    assert not result.success
    assert "configuration changed" in result.metadata["error"]
    assert calls == 1
    assert checkpoint_path.read_bytes() == checkpoint
    assert store.source_path(result.metadata["source"]).read_text() == "evidence"


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
@pytest.mark.parametrize("action", ["create", "update"])
@pytest.mark.parametrize("restart", [False, True])
async def test_repeated_pending_write_stays_failed_until_indexes_recover(
    runtime,
    experience,
    reject_payload,
    monkeypatch,
    app_factory,
    action,
    restart,
):
    app, store, scope = runtime
    if action == "create":
        arguments = {**experience, "content": "unembeddable-payload"}
    else:
        original = await store.apply_mutation("create", experience, scope)
        arguments = {
            "path": original["path"],
            "expected_version": original["version"],
            "operation_key": "pending-update",
            "changes": {"content": "unembeddable-payload"},
        }
    backend = type(store.views["when_to_use"].embedding_store)
    get_embeddings = backend.get_embeddings
    reject_payload(store.views["when_to_use"].embedding_store)
    job = app.context.jobs[f"procedure_{action}"]
    failed = await job(**arguments, _procedure_scope=scope)
    assert not failed.success
    if restart:
        workspace = store.workspace_path
        source = failed.metadata["path"]
        saved_source = (await store.read_experience(source))["experience"]["sources"][0]
        await app.close()
        app, store = await app_factory(workspace=workspace)
        scope = store.open_scope(saved_source, 0)
        job = app.context.jobs[f"procedure_{action}"]
    for _ in range(2):
        repeated = await job(**arguments, _procedure_scope=scope)
        assert not repeated.success
        assert repeated.metadata["status"] == "index_pending"
        assert repeated.metadata["path"] == failed.metadata["path"]
        assert repeated.metadata["version"] == failed.metadata["version"]
        assert repeated.metadata["error"]
        assert "new operation_key" in repeated.metadata["repair"]
        assert not (await store.search("repair", "when_to_use"))["index_complete"]
    monkeypatch.setattr(backend, "get_embeddings", get_embeddings)
    recovered = await job(**arguments, _procedure_scope=scope)
    assert recovered.success, recovered.answer
    assert recovered.metadata["status"] == "indexed"
    assert (await job(**arguments, _procedure_scope=scope)).metadata == recovered.metadata
    for view in store.views.values():
        chunks = list(view.file_chunks.values())
        assert len(chunks) == 1
        assert chunks[0].metadata["version"] == recovered.metadata["version"]


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
@pytest.mark.parametrize("restart", [False, True])
async def test_repair_intent_recovers_without_replaying_old_payload(
    runtime,
    experience,
    reject_payload,
    monkeypatch,
    stage,
    restart,
    app_factory,
):
    app, store, scope = runtime
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
    if restart:
        await app.close()
        _, store = await app_factory(workspace=store.workspace_path)
        scope = store.open_scope("session/trajectories/input.log", 0)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_completed_history_is_not_read_on_search_or_write(
    runtime,
    experience,
    monkeypatch,
    app_factory,
    restart,
):
    app, store, scope = runtime
    source = "session/trajectories/input.log"
    current = await store.apply_mutation("create", experience, scope)
    for number in range(32):
        current = await store.apply_mutation(
            "update",
            {
                "path": current["path"],
                "expected_version": current["version"],
                "operation_key": f"update-{number}",
                "changes": {"content": f"repair index revision {number}"},
            },
            store.open_scope(source, number % 3),
        )
    root = store.state_root / "operations"
    expected = {segment: store.operation_records(source, segment) for segment in (None, 0, 1, 2)}
    reads = []
    original = store_module.load_json

    def track(path):
        if path.parent == root:
            reads.append(path.name)
        return original(path)

    monkeypatch.setattr(store_module, "load_json", track)
    if restart:
        await app.close()
        _, store = await app_factory(workspace=store.workspace_path)
        scope = store.open_scope(source, 0)
        # Indexed journals are decoded only once during startup.
        assert len(reads) == len(set(reads)) == 33
        reads.clear()
    for segment, records in expected.items():
        assert store.operation_records(source, segment) == records
    assert store.operation_records("unknown") == []
    assert all("document" not in receipt for receipt in store._operations.values())
    for _ in range(3):
        assert (await store.search("repair", "when_to_use"))["index_complete"]
    await store.reconcile(source=source, segment=0)
    await store.apply_mutation(
        "update",
        {
            "path": current["path"],
            "expected_version": current["version"],
            "operation_key": "new-update",
            "changes": {"content": "repair latest index"},
        },
        scope,
    )
    assert not reads
    await store.apply_mutation("create", experience, scope)
    assert len(reads) == 1  # An idempotent retry reads just its own journal.


@pytest.mark.asyncio
async def test_recovery_reads_only_pending_history(runtime, experience, reject_payload, monkeypatch):
    _, store, scope = runtime
    await store.apply_mutation("create", experience, scope)
    reject_payload(store.views["when_to_use"].embedding_store)
    with pytest.raises(RuntimeError, match="indexes pending"):
        await store.apply_mutation(
            "create",
            {**experience, "filename": "pending.json", "operation_key": "bad", "content": "unembeddable-payload"},
            scope,
        )
    root = store.state_root / "operations"
    pending = next(path.name for path in root.glob("*.json") if load_json(path)["status"] == "committed")
    original = store_module.load_json
    reads = []

    def track(path):
        if path.parent == root:
            reads.append(path.name)
        return original(path)

    monkeypatch.setattr(store_module, "load_json", track)
    for _ in range(3):
        result = await store.search("repair", "when_to_use_content")
        assert not result["index_complete"]
        assert len(result["results"]) == 1
    assert reads == [pending] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_alias_correction_supersedes_pending_payload(runtime, experience, reject_payload, app_factory, restart):
    app, store, scope = runtime
    reject_payload(store.views["when_to_use"].embedding_store)
    with pytest.raises(RuntimeError, match="indexes pending"):
        await store.apply_mutation("create", {**experience, "content": "unembeddable-payload"}, scope)
    saved = await store.read_experience("procedures/repair-search-index.json")
    corrected = await store.apply_mutation(
        "update",
        {
            "path": "./" + saved["path"],
            "expected_version": saved["version"],
            "operation_key": "alias-repair",
            "changes": {"content": "valid repair advice"},
        },
        scope,
    )
    assert corrected["path"] == saved["path"]
    if restart:
        await app.close()
        _, store = await app_factory(workspace=store.workspace_path)
    await store.reconcile()
    records = store.operation_records("session/trajectories/input.log")
    assert {record["path"] for record in records} == {saved["path"]}
    assert {record["operation_key"]: record["status"] for record in records} == {
        experience["operation_key"]: "superseded",
        "alias-repair": "indexed",
    }
    assert (await store.search("index", "when_to_use_content"))["index_complete"]
    for index in store.views.values():
        assert [node.path for node in await index.get_nodes()] == [saved["path"]]


@pytest.mark.asyncio
async def test_legacy_alias_journal_recovers_and_remains_retryable(runtime, experience, app_factory):
    from reme_procedure_memory.checkpoint import atomic_json, digest, json_text

    app, store, scope = runtime
    created = await store.apply_mutation("create", experience, scope)
    arguments = {
        "path": "./" + created["path"],
        "expected_version": created["version"],
        "operation_key": "legacy-update",
        "changes": {"content": "legacy repair advice"},
    }
    updated = await store.apply_mutation("update", arguments, scope)
    # Simulate a pre-fix journal interrupted after committing the file.
    for journal in (store.state_root / "operations").glob("*.json"):
        record = load_json(journal)
        if record["operation_key"] == arguments["operation_key"]:
            record["path"] = arguments["path"]
            record["signature"] = digest(json_text({"action": "update", "arguments": arguments}))
            record["status"] = "committed"
            atomic_json(journal, record)
    await app.close()
    _, reopened = await app_factory(workspace=store.workspace_path)
    scope = reopened.open_scope("session/trajectories/input.log", 0)
    for path in (arguments["path"], created["path"]):
        assert await reopened.apply_mutation("update", {**arguments, "path": path}, scope) == updated
    with pytest.raises(ValueError, match="different arguments"):
        await reopened.apply_mutation("update", {**arguments, "path": "procedures/another-file.json"}, scope)
    await reopened.reconcile()
    assert (await reopened.search("index", "when_to_use_content"))["index_complete"]
    for index in reopened.views.values():
        assert [node.path for node in await index.get_nodes()] == [created["path"]]


@pytest.mark.asyncio
async def test_legacy_pending_alias_is_superseded_after_restart(runtime, experience, app_factory, reject_payload):
    from reme_procedure_memory.checkpoint import atomic_json, digest, json_text

    app, store, scope = runtime
    created = await store.apply_mutation("create", experience, scope)
    reject_payload(store.views["when_to_use"].embedding_store)
    arguments = {
        "path": "./" + created["path"],
        "expected_version": created["version"],
        "operation_key": "legacy-pending",
        "changes": {"content": "unembeddable-payload"},
    }
    with pytest.raises(RuntimeError, match="indexes pending"):
        await store.apply_mutation("update", arguments, scope)
    for journal in (store.state_root / "operations").glob("*.json"):
        record = load_json(journal)
        if record["operation_key"] == arguments["operation_key"]:
            record["path"] = arguments["path"]
            record["signature"] = digest(json_text({"action": "update", "arguments": arguments}))
            atomic_json(journal, record)
    await app.close()
    _, reopened = await app_factory(workspace=store.workspace_path)
    saved = await reopened.read_experience(created["path"])
    scope = reopened.open_scope("session/trajectories/input.log", 0)
    corrected = await reopened.apply_mutation(
        "update",
        {
            "path": saved["path"],
            "expected_version": saved["version"],
            "operation_key": "repair-legacy",
            "changes": {"content": "valid replacement"},
        },
        scope,
    )
    await reopened.reconcile()
    old = await reopened.apply_mutation("update", arguments, scope)
    assert old["path"] == corrected["path"] == created["path"]
    assert old["status"] == "superseded"
    assert old["superseded_by"] == "repair-legacy"
    assert (await reopened.search("index", "when_to_use_content"))["index_complete"]
