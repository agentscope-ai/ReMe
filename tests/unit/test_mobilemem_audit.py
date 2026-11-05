"""Failure-path receipts and cost accounting, without provider or server calls."""

# pylint: disable=missing-function-docstring,protected-access
import asyncio
import json
from types import SimpleNamespace

import pytest

from benchmark.mobilemem import run
from benchmark.mobilemem._audit import ImageInputError, error_record, mean_tokens, response_record
from reme.schema import Response


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(name="flow")
def fixture_flow(tmp_path, monkeypatch):
    scenes = [
        {
            "id": "uuid_0",
            "image_root": tmp_path,
            "sessions": [{"session_id": "s0", "event_start_time": "2025-01-01T09:00:00"}],
            "questions": [
                {"question_id": "q0", "question": "What?", "answer": "answer", "question_type": "visual_reasoning"},
            ],
        },
    ]
    state = SimpleNamespace(
        scenes=scenes,
        apps=[],
        fail=None,
        direct=False,
        cancel=False,
        cleanup_fail=False,
        startup_fail=False,
        constructor_fail=False,
        calls=[],
    )

    class FakeApp:
        """Mock one owned candidate or Judge application."""

        def __init__(self, *, judge):
            if state.constructor_fail and not judge:
                raise ValueError("candidate construction failed")
            self.judge, self.closed = judge, False
            state.apps.append(self)

        async def start(self):
            if state.startup_fail and not self.judge:
                raise ValueError("candidate startup failed")

        async def close(self):
            self.closed = True
            if state.cleanup_fail and not self.judge:
                raise ValueError("candidate cleanup failed")

        async def run_job(self, name, **kwargs):
            identity = kwargs.get("session_id", kwargs.get("query", ""))
            state.calls.append((name, kwargs))
            if name == "answer_judge":
                assert run._CURRENT.get() is None  # Judge traffic cannot join candidate cost.
            elif name != "index_update":
                run._CURRENT.get().append(run.normalized_usage({"prompt_tokens": 100, "completion_tokens": 20}))
            await asyncio.sleep(0)  # Force interleaving of independently scoped requests.
            if state.fail in (name, identity):
                if state.cancel:
                    raise asyncio.CancelledError("cancelled offline call")
                if state.direct:
                    raise ValueError("direct offline failure")
                return Response(success=False, answer="HTTP 400 input length exceeds limit")
            if name == "answer_judge":
                return Response(answer="1", metadata={"llm_judge_score": 1.0, "parse_default_zero": False})
            # This portable counter is deliberately different: never sum it with HTTP receipts.
            return Response(answer="answer", metadata={"usage": {"total_tokens": 999999}})

    monkeypatch.setattr(run, "Application", FakeApp)
    monkeypatch.setattr(run, "candidate_config", lambda *_: {"judge": False})
    monkeypatch.setattr(run, "judge_config", lambda *_: {"judge": True})
    monkeypatch.setattr(run, "judge_settings", lambda _: {"model": "offline", "protocol": "offline"})
    monkeypatch.setattr(run, "session_messages", lambda *_: [{"original": "unchanged"}])
    monkeypatch.setattr(run, "observe_usage", lambda _: None)
    monkeypatch.setattr(run, "observe_completion", lambda _: None)

    async def execute():
        complete = await run.run(scenes, tmp_path / "run", "candidate", 4, 16)
        return complete, read(tmp_path / "run/summary.json")

    state.execute = execute
    state.directory = tmp_path / "run/users/uuid_0"
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
async def test_failed_second_memory_keeps_both_requests(flow, direct):
    flow.scenes[0]["sessions"].append({"session_id": "s1", "event_start_time": "2025-01-02T09:00:00"})
    flow.fail, flow.direct = "s1", direct
    complete, summary = await flow.execute()
    assert not complete and summary["completed"] == 0
    assert summary["memory_usage"]["total_tokens"] == summary["token_usage"]["total_tokens"] == 240
    assert summary["token_usage"]["records"] == 2
    assert not (flow.directory / "memory_usage.json").exists()
    response = read(flow.directory / "sessions/00001.json")["response"]
    assert response is None if direct else response["success"] is False
    error = read(flow.directory / "error.json")
    assert (error["stage"], error["user_id"], error["session_id"]) == ("memory", "uuid_0", "s1")
    assert all(app.closed for app in flow.apps)


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
async def test_failed_answer_keeps_memory_and_answer_cost_once(flow, direct):
    flow.fail, flow.direct = "agentic_answer", direct
    complete, summary = await flow.execute()
    assert not complete and summary["completed"] == 0 and summary["metrics"]["llm_judge"] is None
    assert summary["token_usage"]["total_tokens"] == 240
    assert summary["memory_usage"]["total_tokens"] == summary["answer_usage"]["total_tokens"] == 120
    assert summary["usage_counts"]["attempted_answers"] == 1
    assert summary["usage_counts"]["successful_answers"] == 0
    assert summary["mean_tokens_per_question"]["total_tokens"] == 240
    assert summary["usage_denominators"] == {"scheduled_users": 1, "expected_questions": 1}
    assert summary["usage_policy"]["version"] == 2
    assert (flow.directory / "memory_usage.json").exists()
    question = flow.directory / "questions/0000"
    receipt = read(question / "answer-response.json")
    assert receipt["usage"]["total_tokens"] == 120
    assert (receipt["response"] is None) is direct
    assert not (question / "answer.json").exists() and not (question / "result.json").exists()
    assert read(question / "error.json")["stage"] == "answer"


@pytest.mark.asyncio
async def test_preprocessing_failure_keeps_earlier_memory_and_image_identity(flow, monkeypatch):
    flow.scenes[0]["sessions"].append({"session_id": "s1", "event_start_time": "2025-01-02T09:00:00"})

    def messages(session, *_):
        if session["session_id"] == "s1":
            try:
                raise FileNotFoundError("missing original image")
            except FileNotFoundError as error:
                raise ImageInputError("image/uid0/photo.png", error) from error
        return []

    monkeypatch.setattr(run, "session_messages", messages)
    _, summary = await flow.execute()
    assert summary["memory_usage"]["total_tokens"] == 120
    receipt = read(flow.directory / "sessions/00001.json")
    assert receipt["response"] is None and receipt["usage"]["records"] == 0
    error = read(flow.directory / "error.json")
    assert error["stage"] == "preprocessing" and error["image_path"] == "image/uid0/photo.png"
    assert [cause["type"] for cause in error["chain"]] == ["ImageInputError", "FileNotFoundError"]
    assert not (flow.directory / "memory_usage.json").exists()


@pytest.mark.asyncio
async def test_index_failure_is_saved_without_complete_memory_marker(flow):
    flow.fail = "index_update"
    _, summary = await flow.execute()
    assert summary["memory_usage"]["total_tokens"] == 120
    assert read(flow.directory / "index-response.json")["response"]["success"] is False
    assert read(flow.directory / "error.json")["stage"] == "index"
    assert not (flow.directory / "memory_usage.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
async def test_judge_failure_preserves_answer_without_double_counting(flow, direct):
    flow.fail, flow.direct = "answer_judge", direct
    complete, summary = await flow.execute()
    assert not complete and summary["token_usage"]["total_tokens"] == 240
    assert summary["answer_usage"]["records"] == 1
    question = flow.directory / "questions/0000"
    assert read(question / "answer-response.json")["response"]["success"] is True
    assert not (question / "answer.json").exists()
    assert not (question / "result.json").exists()
    assert read(question / "error.json")["stage"] == "judge"
    assert (read(question / "judge.json") is None) is direct


@pytest.mark.asyncio
async def test_all_success_old_payloads_metrics_and_usage_stay_compatible(flow):
    flow.scenes[0]["sessions"].append({"session_id": "s1", "event_start_time": "2025-01-02T09:00:00"})
    complete, summary = await flow.execute()
    assert complete and summary["completed"] == summary["expected"] == 1
    assert summary["metrics"] == {"f1": 1.0, "bleu_1": 1.0, "llm_judge": 1.0}
    assert summary["token_usage"]["total_tokens"] == 360
    assert summary["memory_usage"]["total_tokens"] == 240
    assert summary["answer_usage"]["total_tokens"] == 120
    assert summary["mean_tokens_per_user"]["total_tokens"] == summary["mean_tokens_per_question"]["total_tokens"] == 360
    assert summary["token_usage"]["cached_input_tokens"] is None
    assert summary["token_usage"]["coverage"]["cached_input_tokens"] == 0
    assert read(flow.directory / "memory_usage.json") == summary["memory_usage"]
    assert [name for name, _ in flow.calls] == [
        "auto_memory",
        "auto_memory",
        "index_update",
        "agentic_answer",
        "answer_judge",
    ]
    question = flow.directory / "questions/0000"
    assert read(question / "answer-response.json")["response"]["success"] is True
    assert not (question / "answer.json").exists()
    result = read(question / "result.json")
    assert result == {
        "question_id": "q0",
        "category": "visual_reasoning",
        "question": "What?",
        "reference": "answer",
        "prediction": "answer",
        "metrics": summary["metrics"],
        "usage": summary["answer_usage"],
        "judge_parse_default_zero": False,
    }


@pytest.mark.asyncio
async def test_concurrent_questions_keep_cost_scopes_and_scheduled_denominator(flow):
    flow.scenes[0]["questions"].append({**flow.scenes[0]["questions"][0], "question_id": "q1"})
    flow.scenes.append({**flow.scenes[0], "id": "uuid_1"})
    complete, summary = await flow.execute()
    assert complete and summary["completed"] == 4
    assert summary["token_usage"]["records"] == 6 and summary["token_usage"]["total_tokens"] == 720
    assert summary["mean_tokens_per_user"]["total_tokens"] == 360
    assert summary["mean_tokens_per_question"]["total_tokens"] == 180


@pytest.mark.asyncio
async def test_failure_means_use_scheduled_not_successful_denominators(flow):
    flow.scenes[0]["questions"].append({**flow.scenes[0]["questions"][0], "question_id": "q1"})
    flow.fail = "agentic_answer"
    _, summary = await flow.execute()
    assert summary["completed"] == 0 and summary["expected"] == 2
    assert summary["token_usage"]["total_tokens"] == 360
    assert summary["mean_tokens_per_question"]["total_tokens"] == 180


@pytest.mark.asyncio
async def test_empty_scheduled_population_preserves_unknown_means(flow):
    flow.scenes.clear()
    _, summary = await flow.execute()
    assert summary["usage_denominators"] == {"scheduled_users": 0, "expected_questions": 0}
    assert all(value is None for value in summary["mean_tokens_per_user"].values())
    assert all(value is None for value in summary["mean_tokens_per_question"].values())
    assert summary["token_usage"]["records"] == 0


@pytest.mark.asyncio
async def test_summary_ignores_legacy_answer_and_result_usage_copies(flow, monkeypatch):
    original = run.close_application

    async def close(app, directory, **kwargs):
        await original(app, directory, **kwargs)
        if app is not None and app.judge:
            for name in ("answer.json", "result.json"):
                path = flow.directory / "questions/0000" / name
                data = read(path) if path.exists() else {"response": {"success": True}}
                data["usage"] = {"total_tokens": 999999}
                run.save(path, data)

    monkeypatch.setattr(run, "close_application", close)
    _, summary = await flow.execute()
    assert summary["token_usage"]["total_tokens"] == 240
    assert summary["answer_usage"]["records"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("constructor", [False, True])
async def test_startup_and_cleanup_failures_are_separate(flow, constructor):
    flow.constructor_fail, flow.startup_fail = constructor, not constructor
    flow.cleanup_fail = True
    complete, summary = await flow.execute()
    assert not complete and summary["token_usage"]["total_tokens"] is None
    assert read(flow.directory / "error.json")["stage"] == "startup"
    assert (flow.directory / "cleanup-error.json").exists() is not constructor
    assert all(app.closed for app in flow.apps)


@pytest.mark.asyncio
async def test_cancelled_request_finalizes_once_and_closes_owned_apps(flow):
    flow.fail, flow.cancel = "auto_memory", True
    with pytest.raises(asyncio.CancelledError):
        await flow.execute()
    receipt = read(flow.directory / "sessions/00000.json")
    assert receipt["response"] is None and receipt["usage"]["total_tokens"] == 120
    assert read(flow.directory / "error.json")["error_type"] == "CancelledError"
    assert all(app.closed for app in flow.apps)
    assert read(flow.directory.parents[1] / "summary.json")["memory_usage"]["total_tokens"] == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_task_joins_sibling_cleanup_before_propagating(cancel):
    started, closed = asyncio.Event(), asyncio.Event()

    async def waiting():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            closed.set()

    async def interrupted():
        await started.wait()
        if cancel:
            raise asyncio.CancelledError()
        raise ValueError("offline task failed")

    with pytest.raises(asyncio.CancelledError if cancel else ValueError):
        await run.gather_tasks([waiting(), interrupted()])
    assert closed.is_set()


def test_error_artifacts_remove_secrets_payloads_and_locals(monkeypatch):
    secret = "offline-sensitive-value"
    monkeypatch.setenv("LLM_API_KEY", secret)
    binary = "YWJj" * 100
    response = Response(
        success=False,
        answer=f"input length error api_key={secret} data:image/png;base64,{binary}",
        metadata={
            "headers": {"X-Private": "hidden-header"},
            "request": {"body": "hidden-body"},
            "credential": {"token": "hidden-credential"},
            "traceback": "hidden-local",
            "status_code": 400,
        },
    )
    try:
        raise ValueError(f"https://user:password@example.invalid/?key={secret} Bearer token-value sk-private-key")
    except ValueError as error:
        record = error_record(error, stage="answer", user_id="uuid_0", question_id="q0", response=response)
    serialized = json.dumps([record, response_record(response)])
    for value in (
        secret,
        binary,
        "hidden-header",
        "hidden-body",
        "hidden-credential",
        "hidden-local",
        "token-value",
        "sk-private-key",
        "user:password",
    ):
        assert value not in serialized
    assert "input length" in record["reason"]
    assert set(record["chain"][0]["traceback"][0]) == {"file", "line", "function"}
    assert ImageInputError("../../private.png", ValueError("bad")).image_path == "[invalid image path]"
    successful = Response(answer="URL https://example.test/ and literal sk-example-text", metadata={"raw": binary})
    assert response_record(successful) == successful.model_dump()


@pytest.mark.parametrize(
    "name",
    ["token", "refresh_token", "id_token", "access_token", "auth_token", "private_key", "private-key", "privateKey"],
)
def test_credentials_are_redacted_in_failed_responses_and_error_chains(name):
    sentinel = "0123456789abcdef" * 4
    usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "cached_input_tokens": 30}
    response = Response(
        success=False,
        answer=f'provider rejection {name}="{sentinel}"',
        metadata={name: sentinel, "usage": usage},
    )
    try:
        try:
            raise ValueError(f"provider {name}: {sentinel}")
        except ValueError as cause:
            raise RuntimeError(f"job {name}='{sentinel}'") from cause
    except RuntimeError as error:
        diagnostic = error_record(error, stage="answer", user_id="uuid_0", question_id="q0", response=response)
    record = response_record(response)
    assert sentinel not in json.dumps([record, diagnostic])
    assert record["metadata"] == {name: "[redacted]", "usage": usage}
    assert record["answer"] == f"provider rejection {name}=[redacted]"
    assert diagnostic["reason"] == record["answer"]
    assert [item["reason"] for item in diagnostic["chain"]] == [
        f"job {name}=[redacted]",
        f"provider {name}: [redacted]",
    ]
    assert diagnostic["stage"] == "answer" and diagnostic["question_id"] == "q0"
    successful = response.model_copy(update={"success": True})
    assert response_record(successful) == successful.model_dump()


@pytest.mark.parametrize(
    "sentinel",
    ["0123456789abcdef" * 4, "a", "success", "answer", "metadata", "usage", "coverage", "records", "input_tokens"],
)
def test_private_key_environment_values_are_redacted_without_changing_usage_schema(monkeypatch, sentinel):
    monkeypatch.setenv("TEST_PRIVATE_KEY", sentinel)
    usage = {"input_tokens": 100, "output_tokens": 20, "records": 1, "coverage": {"input_tokens": 1}}
    response = Response(
        success=False,
        answer=f"provider returned {sentinel}",
        metadata={"usage": usage, "private_key": "PRIVATE_SENTINEL", f"debug {sentinel}": f"detail {sentinel}"},
    )
    record = response_record(response)
    assert set(record) == {"success", "answer", "metadata"}
    assert Response.model_validate(record).success is False
    assert record["answer"] == "provider returned [redacted]"
    assert record["metadata"] == {
        "usage": usage,
        "private_key": "[redacted]",
        "debug [redacted]": "detail [redacted]",
    }
    diagnostic = error_record(RuntimeError(f"provider returned {sentinel}"), stage="startup")
    assert diagnostic["reason"] == diagnostic["chain"][0]["reason"] == "provider returned [redacted]"


@pytest.mark.parametrize("header", ["Cookie", "Set-Cookie"])
def test_failed_response_redacts_whole_cookie_header_without_changing_ordinary_prose(header):
    response = Response(
        success=False,
        answer=f"{header}: session=opaque; other=secondsecret\nThe cookie parser failed with status 400.",
    )
    record = response_record(response)
    assert "opaque" not in record["answer"] and "secondsecret" not in record["answer"]
    assert "The cookie parser failed with status 400." in record["answer"]


@pytest.mark.asyncio
async def test_http_observer_includes_compressor_subsets_and_null_coverage_without_duplicate_wrapping():
    async def send(request):
        await asyncio.sleep(0)
        return SimpleNamespace(status_code=200, is_stream_consumed=True, json=lambda: {"usage": request})

    clients = [SimpleNamespace(send=send) for _ in range(3)]
    app = SimpleNamespace(
        context=SimpleNamespace(
            components={
                "as_llm": {
                    name: SimpleNamespace(model=SimpleNamespace(client=SimpleNamespace(_client=client)))
                    for name, client in zip(("default", "bench", "compressor"), clients)
                },
            },
        ),
    )
    run.observe_usage(app)
    run.observe_usage(app)

    async def scoped(amount):
        with run.usage_scope() as records:
            await clients[0].send(
                {"prompt_tokens": amount, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 30}},
            )
            await clients[2].send(
                {"prompt_tokens": 50, "completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 4}},
            )
        return run.aggregate_usage(records)

    first, second = await asyncio.gather(scoped(100), scoped(200))
    assert first["records"] == second["records"] == 2
    assert (first["total_tokens"], second["total_tokens"]) == (180, 280)
    assert first["cached_input_tokens"] == 30 and first["noncached_input_tokens"] == 70
    assert first["coverage"]["cached_input_tokens"] == 1 and first["reasoning_tokens"] == 4
    assert mean_tokens(first, 0)["total_tokens"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_job,completed,cost", [("auto_memory", 0, 120), (None, 1, 240)])
async def test_cleanup_failure_preserves_primary_outcome(flow, failed_job, completed, cost):
    flow.fail, flow.cleanup_fail = failed_job, True
    complete, summary = await flow.execute()
    assert not complete and summary["completed"] == completed
    assert summary["cleanup_error_count"] == 1 and summary["token_usage"]["total_tokens"] == cost
    assert read(flow.directory / "cleanup-error.json")["stage"] == "cleanup"
    if failed_job:
        assert read(flow.directory / "error.json")["stage"] == "memory"
    else:
        assert summary["metrics"]["llm_judge"] == 1.0
