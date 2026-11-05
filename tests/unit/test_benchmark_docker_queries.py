"""Host-only query answers never participate in candidate serialization."""

# pylint: disable=missing-function-docstring,protected-access

import json
import threading
from unittest.mock import AsyncMock

import pytest

from deploy.benchmark_docker import EvaluationQuery, ImageCandidate
from deploy.benchmark_docker.docker_sandbox import DockerReMeSandbox


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "as_mapping"), [("run_query", False), ("run_queries", True)])
async def test_query_upload_ignores_uncopyable_golden_answer(monkeypatch, method, as_mapping):
    case = DockerReMeSandbox(
        case_id="query",
        container_id="unused",
        candidate=ImageCandidate("unused"),
        workspace=AsyncMock(),
        config="default",
        plugins=(),
        command_timeout=5,
        environment_names=[],
    )
    arguments = {"limit": 3}
    golden_answer = threading.Lock()
    query = EvaluationQuery("q1", "Question?", golden_answer, arguments, "custom_answer")
    if as_mapping:
        query = {
            "query_id": "q1",
            "question": "Question?",
            "golden_answer": golden_answer,
            "answer_arguments": arguments,
            "answer_job": "custom_answer",
            "rubric": golden_answer,
        }
    result = {"answer": "candidate answer"}
    worker = AsyncMock(return_value={"queries": [result]})
    monkeypatch.setattr(case, "_run_worker_request", worker)

    if method == "run_query":
        assert await case.run_query(query) == result
    else:
        assert await case.run_queries([query]) == {"queries": [result]}

    mode, payload = worker.await_args.args
    assert mode == ("query" if method == "run_query" else "queries")
    assert payload == {
        "queries": [
            {
                "query_id": "q1",
                "question": "Question?",
                "answer_arguments": {"limit": 3},
                "answer_job": "custom_answer",
            },
        ],
    }
    assert payload["queries"][0]["answer_arguments"] is not arguments
    # A host-only lock is neither deep-copied nor included in the JSON request.
    json.dumps(payload)
    assert not golden_answer.locked()
