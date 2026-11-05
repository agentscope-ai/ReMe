"""Offline failure, accounting and artifact contracts for the MemLens runner."""

# pylint: disable=missing-function-docstring,protected-access
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmark.memlens import run as runner
from benchmark.memlens.artifacts import error_details, judge_settings, model_settings, response_record, safe_value
from reme.schema import Response


@pytest.fixture(autouse=True)
def memlens_judge_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins/memlens-judge/src"))


@pytest.fixture(name="scene_run")
def fixture_scene_run(tmp_path, monkeypatch):
    import reme

    scene = {
        "id": "scene-1",
        "sessions": [{"id": "session-1", "timestamp": "2026-01-01"}],
        "questions": [
            {
                "id": "question-1",
                "text": "Question",
                "question_date": "2026-01-02",
                "answer": "Answer",
                "category": "information_extraction",
            },
        ],
    }
    app = SimpleNamespace(start=AsyncMock(), close=AsyncMock(), run_job=AsyncMock())
    judge = SimpleNamespace(run_job=AsyncMock(return_value=Response(metadata={"llm_judge_score": 1})))
    monkeypatch.setattr(reme, "Application", lambda **_: app)
    monkeypatch.setattr(runner, "app_config", lambda *_, **__: {})
    monkeypatch.setattr(runner, "observe_usage", lambda _: None)
    monkeypatch.setattr(runner, "session_messages", lambda _: [{"role": "user", "content": "Text"}])
    case = tmp_path / "reme_image/scenes/0000-scene-1"

    async def evaluate():
        return await runner.evaluate_scene(scene, 0, "candidate", tmp_path, judge, asyncio.Semaphore(1))

    return scene, app, judge, case, evaluate


def measured(tokens=120):
    return runner.normalized_usage({"input_tokens": tokens - 20, "output_tokens": 20})


@pytest.mark.asyncio
async def test_index_failure_retains_completed_session_usage(scene_run):
    _, app, _, case, evaluate = scene_run

    async def job(name, **_):
        if name == "auto_memory":
            runner._CURRENT.get().append(measured())
            return Response(answer="Memory")
        return Response(success=False, answer="Index unavailable")

    app.run_job.side_effect = job
    results, memory, answers, status = await evaluate()
    assert results == answers == []
    assert runner.total_usage(memory)["total_tokens"] == 120
    assert json.loads((case / "status.json").read_text()) == status
    assert status["stage"] == "index_update" and status["scene_id"] == "scene-1"
    assert "response detail: Index unavailable" in status["error"]["message"]
    assert status["error_type"] == "RuntimeError"
    assert status["memory_complete"] is False
    assert len(list((case / "ingestion/attempt-001/sessions").glob("*.json"))) == 1
    artifact = json.loads((case / "ingestion/attempt-001/diagnostics/index-response.json").read_text())
    assert artifact["response"]["success"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
async def test_failed_memory_keeps_its_successful_requests_and_diagnostics(scene_run, direct):
    scene, app, _, case, evaluate = scene_run
    scene["sessions"].append({"id": "session-2", "timestamp": "2026-01-02"})

    async def job(*_, session_id, **__):
        runner._CURRENT.get().append(measured())
        if session_id == "session-1":
            return Response(answer="Memory")
        if direct:
            try:
                raise ValueError("Original provider failure")
            except ValueError as exc:
                raise RuntimeError("Memory failed") from exc
        return Response(success=False, metadata={"error": "Memory failed"})

    app.run_job.side_effect = job
    _, memory, _, _ = await evaluate()
    assert runner.total_usage(memory)["total_tokens"] == 240
    status = json.loads((case / "status.json").read_text())
    assert status["stage"] == "auto_memory" and status["session_id"] == "session-2"
    assert status["memory_complete"] is False
    assert (case / "ingestion/attempt-001/sessions/00000.json").exists()
    assert not (case / "ingestion/attempt-001/sessions/00001.json").exists()
    assert json.loads((case / "ingestion/attempt-001/diagnostics/00001-usage.json").read_text())["total_tokens"] == 120
    if direct:
        assert status["error"]["cause"]["message"] == "Original provider failure"
        assert status["error"]["traceback"] and "locals" not in json.dumps(status)
    else:
        artifact = json.loads((case / "ingestion/attempt-001/diagnostics/00001-response.json").read_text())
        assert artifact["response"]["success"] is False
    app.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_job", ["agentic_answer", "judge"])
async def test_failed_answer_or_judge_keeps_answer_usage_once(scene_run, failed_job):
    _, app, judge, case, evaluate = scene_run

    async def job(name, **_):
        if name in {"auto_memory", "agentic_answer"}:
            runner._CURRENT.get().append(measured())
        return Response(answer="Answer", success=name != failed_job)

    app.run_job.side_effect = job
    if failed_job == "judge":
        judge.run_job.return_value = Response(success=False, metadata={"error": "Judge failed"})
    results, memory, answers, _ = await evaluate()
    assert results == [] and len(memory) == len(answers) == 1
    assert runner.total_usage(answers)["total_tokens"] == 120
    status = json.loads((case / "status.json").read_text())
    assert status["question_id"] == "question-1" and status["stage"] == failed_job
    assert status["memory_complete"] is True
    path = "judge-response.json" if failed_job == "judge" else "answer-response.json"
    assert json.loads((case / "questions/0000/diagnostics" / path).read_text())["response"]["success"] is False
    assert (case / "questions/0000/answer.json").exists() is (failed_job == "judge")
    assert json.loads((case / "questions/0000/diagnostics/answer-usage.json").read_text())["total_tokens"] == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["preprocess", "start", "cleanup", "cancel"])
async def test_direct_lifecycle_failures_are_recorded(scene_run, monkeypatch, stage):
    _, app, _, case, evaluate = scene_run
    app.run_job.return_value = Response(answer="Answer")
    if stage == "start":
        app.start.side_effect = RuntimeError("startup failed")
    elif stage == "cleanup":
        app.close.side_effect = RuntimeError("cleanup failed")
    elif stage == "cancel":
        app.run_job.side_effect = asyncio.CancelledError()
    else:

        def broken(_):
            raise ValueError("Invalid image: release_images/missing.png")

        monkeypatch.setattr(runner, "session_messages", broken)
    if stage == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await evaluate()
    else:
        await evaluate()
    status = json.loads((case / "status.json").read_text())
    assert status["complete"] is False
    assert status["stage"] == ("auto_memory" if stage == "cancel" else stage)
    assert status["error"]["type"] == (
        "CancelledError" if stage == "cancel" else "RuntimeError" if stage in {"start", "cleanup"} else "ValueError"
    )
    if stage == "preprocess":
        assert status["session_id"] == "session-1" and "missing.png" in status["error"]["message"]
        assert status["error"]["traceback"]


@pytest.mark.asyncio
async def test_observer_separates_concurrent_scopes_including_compression_and_unknown_counters():
    def response(usage, status=200):
        return SimpleNamespace(status_code=status, is_stream_consumed=True, json=lambda: {"usage": usage})

    client = SimpleNamespace(send=AsyncMock(side_effect=response))
    compressor = SimpleNamespace(send=AsyncMock(side_effect=response))
    app = SimpleNamespace(
        context=SimpleNamespace(
            components={
                "as_llm": {
                    "default": SimpleNamespace(model=SimpleNamespace(client=SimpleNamespace(_client=client))),
                    "compressor": SimpleNamespace(model=SimpleNamespace(client=SimpleNamespace(_client=compressor))),
                },
            },
        ),
    )
    runner.observe_usage(app)

    async def request(tokens):
        with runner.collect_usage() as records:
            await client.send({"prompt_tokens": tokens, "completion_tokens": 20})
            await asyncio.sleep(0)
            await compressor.send({})
        return records

    first, second = await asyncio.gather(request(100), request(200))
    assert runner.total_usage(first)["total_tokens"] == 120
    assert runner.total_usage(second)["total_tokens"] == 220
    assert runner.total_usage(first)["records"] == 2
    assert runner.total_usage(first)["coverage"]["total_tokens"] == 1
    assert runner.total_usage(first)["cached_input_tokens"] is None
    assert runner._CURRENT.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
async def test_failure_artifacts_redact_secrets_and_request_payloads(scene_run, monkeypatch, direct):
    _, app, _, case, evaluate = scene_run
    secret = "ENV_SECRET_SENTINEL_314159"
    monkeypatch.setenv("LLM_API_KEY", secret)
    detail = (
        f"provider failed {secret}; api_key=INLINE_SENTINEL "
        "Bearer AUTH_SENTINEL https://USER_SENTINEL:PASS_SENTINEL@provider.test/v1?key=URL_SENTINEL "
        f"data:image/png;base64,{'QmFzZTY0' * 20}"
    )
    if direct:
        app.run_job.side_effect = RuntimeError(detail)
    else:
        app.run_job.return_value = Response(
            success=False,
            metadata={
                "error": detail,
                "request_headers": {"arbitrary": "HEADER_SENTINEL"},
                "config": {"arbitrary": "CONFIG_SENTINEL"},
                "source": {"type": "base64", "data": "SHORT_IMAGE_SENTINEL"},
            },
        )
    await evaluate()
    artifacts = "".join(path.read_text() for path in case.rglob("*.json"))
    for value in (
        secret,
        "INLINE_SENTINEL",
        "AUTH_SENTINEL",
        "USER_SENTINEL",
        "PASS_SENTINEL",
        "URL_SENTINEL",
        "HEADER_SENTINEL",
        "CONFIG_SENTINEL",
        "SHORT_IMAGE_SENTINEL",
        "QmFzZTY0",
    ):
        assert value not in artifacts
    assert "provider failed" in artifacts and "[REDACTED" in artifacts


@pytest.mark.asyncio
async def test_successful_predictions_and_references_remain_verbatim(scene_run):
    scene, app, judge, case, evaluate = scene_run
    answer = "token=ordinary-text https://example.test/path?a=b data:image/png;base64," + "QmFzZTY0" * 20
    scene["questions"][0]["answer"] = answer
    app.run_job.return_value = Response(answer=answer)
    results, *_ = await evaluate()
    assert results[0]["prediction"] == results[0]["reference"] == answer
    assert judge.run_job.call_args.kwargs["prediction"] == answer
    for name in ("answer.json", "result.json"):
        assert json.loads((case / "questions/0000" / name).read_text())["prediction"] == answer
    assert json.loads((case / "status.json").read_text())["memory_complete"] is True


@pytest.mark.asyncio
async def test_cancellation_finalizes_observed_requests_before_propagating(scene_run):
    _, app, _, case, evaluate = scene_run

    async def canceled(*_, **__):
        runner._CURRENT.get().append(measured())
        raise asyncio.CancelledError()

    app.run_job.side_effect = canceled
    with pytest.raises(asyncio.CancelledError):
        await evaluate()
    assert json.loads((case / "status.json").read_text())["memory_usage"]["total_tokens"] == 120
    assert json.loads((case / "ingestion/attempt-001/diagnostics/00000-usage.json").read_text())["total_tokens"] == 120
    app.close.assert_awaited_once()


@pytest.mark.parametrize("field", ["cookie", "set-cookie", "private_key", "id_token", "refresh_token", "token"])
def test_diagnostic_credentials_are_redacted_without_removing_counters(field):
    diagnostic = safe_value({field: "DIAGNOSTIC_SECRET_SENTINEL", "input_tokens": 120, "output_tokens": 20})
    assert diagnostic == {field: "[REDACTED]", "input_tokens": 120, "output_tokens": 20}
    assert "DIAGNOSTIC_SECRET_SENTINEL" not in json.dumps(
        error_details(RuntimeError(f"{field}: DIAGNOSTIC_SECRET_SENTINEL")),
    )


@pytest.mark.parametrize(
    "secret",
    ["response", "success", "answer", "metadata", "usage", "coverage", "records", "input_tokens"],
)
def test_failed_response_schema_survives_environment_name_collision(monkeypatch, secret):
    monkeypatch.setenv("TEST_PRIVATE_KEY", secret)
    usage = {"input_tokens": 100, "output_tokens": 20, "records": 1, "coverage": {"input_tokens": 1}}
    raw = Response(
        success=False,
        answer=f"Provider unavailable: {secret}",
        metadata={"usage": usage, "private_key": "PRIVATE_SENTINEL", f"debug {secret}": f"detail {secret}"},
    )
    response = json.loads(json.dumps(response_record(raw)))
    assert Response.model_validate(response).success is False
    assert response["answer"] == "Provider unavailable: [REDACTED]"
    assert response["metadata"] == {
        "usage": usage,
        "private_key": "[REDACTED]",
        "debug [REDACTED]": "detail [REDACTED]",
    }


@pytest.mark.parametrize("secret", ["a", "model", "temperature"])
def test_model_snapshot_preserves_whitelisted_schema(monkeypatch, secret):
    monkeypatch.setenv("LLM_API_KEY", secret)
    spec = {"backend": "agentscope", "model": "candidate", "parameters": {"temperature": 0, "max_tokens": 256}}
    config = SimpleNamespace(components={"as_llm": {"default": SimpleNamespace(model_dump=lambda: spec)}})
    snapshot = model_settings(config)["default"]
    assert snapshot["backend"] == "agentscope" and snapshot["model"] == "candidate"
    assert snapshot["parameters"] == {"temperature": 0, "max_tokens": 256}


@pytest.mark.asyncio
@pytest.mark.parametrize("secret", ["a", "stage"])
@pytest.mark.parametrize("outcome", ["complete", "index_update", "cleanup"])
async def test_run_summary_keeps_usage_and_independent_completion(scene_run, monkeypatch, secret, outcome):
    import reme
    from reme.schema import ApplicationConfig

    scene, app, judge, case, _ = scene_run
    root = case.parents[2]
    monkeypatch.setenv("LLM_API_KEY", secret)
    monkeypatch.setattr(runner, "app_config", lambda workspace, *_, **__: {"workspace_dir": str(workspace)})
    monkeypatch.setattr(runner, "load_memlens", lambda *_: [scene])
    judge.start, judge.close = AsyncMock(), AsyncMock()
    judge.config = ApplicationConfig(workspace_dir=root / "judge")
    applications = iter([judge, app])
    monkeypatch.setattr(reme, "Application", lambda **_: next(applications))

    async def job(name, **_):
        if name != "index_update":
            runner._CURRENT.get().append(measured())
        return Response(answer="Answer", success=name != outcome)

    app.run_job.side_effect = job
    if outcome == "cleanup":
        app.close.side_effect = RuntimeError("Close failed")
    summary = await runner.run(
        {
            "dataset": str(root / "unused"),
            "output": str(root / "output"),
            "query_concurrency": 1,
            "scenario_concurrency": 1,
            "unexpected": "RAW_CONFIG_SENTINEL",
        },
        "candidate",
        "128k",
    )
    group = root / "output/memlens-128k/candidate"
    status = json.loads((group / "reme_image/scenes/0000-scene-1/status.json").read_text())
    complete = outcome == "complete"
    memory_complete = outcome != "index_update"
    assert status["stage"] == outcome and status["scene_id"] == "scene-1"
    assert status["memory_complete"] is memory_complete and status["complete"] is complete
    assert status["memory_usage"]["input_tokens"] == 100
    assert summary["scored_questions"] == summary["memory_complete_scenes"] == int(memory_complete)
    assert summary["token_usage"]["total_tokens"] == (240 if memory_complete else 120)
    assert summary["cleanup_error_count"] == int(outcome == "cleanup")
    assert summary["completed_scenes"] == int(complete)
    assert summary["usage_policy_version"] == "successful-http-requests-v2"
    assert json.loads((group / "summary.json").read_text()) == summary
    assert json.loads((group / "status.json").read_text())["complete"] is complete
    settings = (group / "settings.json").read_text()
    assert "RAW_CONFIG_SENTINEL" not in settings
    assert json.loads(settings)["judge"]["protocol_id"] == "memlens-task-binary-api-qwen-v1"
    if outcome == "cleanup":
        assert status["error_type"] == status["error"]["type"] == "RuntimeError"
        assert status["error"]["message"] == "Close failed"


@pytest.mark.asyncio
async def test_judge_snapshot_uses_validated_plugin_and_environment_config(tmp_path, monkeypatch):
    from importlib.metadata import EntryPoint

    from reme import Application
    from reme.plugin import PluginManager, _load_plugin

    plugin = _load_plugin(
        "memlens-judge",
        EntryPoint(
            name="memlens-judge",
            value="judge_memlens",
            group="reme.plugins",
        ),
    )
    manager = PluginManager([plugin])
    monkeypatch.setattr(PluginManager, "discover", classmethod(lambda cls, _: manager))
    monkeypatch.setenv("JUDGE_MODEL_NAME", "judge-override-for-test")
    monkeypatch.setenv("LLM_API_KEY", "SNAPSHOT_SECRET_SENTINEL")
    monkeypatch.setenv("LLM_BASE_URL", "https://private.test/v1?key=URL_SENTINEL")
    inputs = runner.app_config(tmp_path / "judge", "candidate-model", judge=True)
    assert inputs["components"] == {}
    app = Application(**inputs)
    snapshot = judge_settings(app)
    model = snapshot["models"]["judge"]
    assert model["model"] == "judge-override-for-test"
    assert model["parameters"] == {
        "temperature": 0,
        "thinking_enable": False,
        "max_tokens": 256,
    }
    assert model["extra_body"] == {"enable_thinking": False}
    assert model["client_kwargs"] == {"timeout": 600, "max_retries": 0}
    assert snapshot["protocol_id"] == "memlens-task-binary-api-qwen-v1"
    assert "SENTINEL" not in json.dumps(snapshot) and "credential" not in json.dumps(snapshot)
    await app.close()


def test_cli_reports_sanitized_direct_failure(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["memlens"])
    monkeypatch.setattr(runner, "run", AsyncMock(side_effect=RuntimeError("provider api_key=CLI_SECRET_SENTINEL")))
    assert runner.main() == 2
    diagnostic = capsys.readouterr().err
    assert "CLI_SECRET_SENTINEL" not in diagnostic
    assert json.loads(diagnostic)["error"]["message"] == "provider api_key=[REDACTED]"
