"""Real Docker smoke test for the reusable LongMemEval sandbox.

This test uses the first LongMemEval item but intentionally ingests only its
first two chronological sessions. Answer quality is outside the contract: the
test verifies the end-to-end container workflow and the analysis archive.

Prerequisites:

* a running Docker daemon;
* ``reme-sandbox-base:agentscope-2.0.4-post1`` built from ``Dockerfile.base``;
* LLM and embedding credentials in the environment or repository ``.env``.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

from dotenv import load_dotenv
import pytest

from benchmark.longmemeval.run import format_messages_for_reme, parse_haystack_date, sessions_sorted_by_time, to_iso
from sandbox import DockerReMeSandboxFactory, EvaluationQuery, JobRequest, SourceCandidate, SourceSnapshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "benchmark/datasets/longmemeval/longmemeval_s_reme_cleaned.json"
SANDBOX_ENV_NAMES = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_BACKEND",
    "LLM_MODEL_NAME",
    "BENCH_MODEL_NAME",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_BACKEND",
    "EMBEDDING_MODEL_NAME",
)


def _archive_members(payload: bytes) -> tuple[set[str], dict[str, bytes]]:
    """Return normalized member names and regular-file payloads."""
    names: set[str] = set()
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            normalized = PurePosixPath(member.name).as_posix().removeprefix("./")
            names.add(normalized.rstrip("/"))
            if member.isfile():
                extracted = archive.extractfile(member)
                assert extracted is not None
                files[normalized] = extracted.read()
    return names, files


@pytest.mark.asyncio
async def test_first_longmemeval_case_evaluation_export(tmp_path):
    """Build once, reuse one query Application, and verify evaluation artifacts."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    missing = [name for name in ("LLM_API_KEY", "EMBEDDING_API_KEY") if not os.environ.get(name)]
    if missing:
        pytest.skip(f"missing sandbox integration credentials: {', '.join(missing)}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    item = dataset[0]
    sessions = sessions_sorted_by_time(item)[:2]
    assert len(sessions) == 2

    env = {name: os.environ[name] for name in SANDBOX_ENV_NAMES if os.environ.get(name)}
    candidate = SourceCandidate(SourceSnapshot.from_directory(PROJECT_ROOT))
    factory = DockerReMeSandboxFactory(candidate, env=env, command_timeout=1800.0)
    case = await factory.create_case(f"longmemeval-{item['question_id']}-smoke")
    archive_path = tmp_path / "evaluation.tar.gz"
    try:
        assert case.backend is not None
        dependency_check = await case.backend.exec_shell(
            [
                case.python,
                "-c",
                "import importlib.util, json; "
                "print(json.dumps({name: importlib.util.find_spec(name) is not None "
                "for name in ('tushare', 'dingtalk_stream', 'pypdf', 'polars')}))",
            ],
        )
        assert dependency_check.ok(), dependency_check.stderr
        assert json.loads(dependency_check.stdout) == {
            "tushare": False,
            "dingtalk_stream": False,
            "pypdf": False,
            "polars": False,
        }

        build_jobs = []
        for _, session_dt, session_id, messages in sessions:
            build_jobs.extend(
                [
                    JobRequest(
                        "auto_memory",
                        {
                            "session_id": session_id,
                            "messages": format_messages_for_reme(messages, session_dt),
                            "date": session_dt.strftime("%Y-%m-%d"),
                        },
                    ),
                    JobRequest("index_update"),
                ],
            )
        build = await case.run_build(build_jobs)
        assert build["success"], build
        await case.commit_memory_history("constructed LongMemEval memory")

        evaluation = await case.run_queries(
            [
                EvaluationQuery(
                    query_id=str(item["question_id"]),
                    question=item["question"],
                    golden_answer=item["answer"],
                    answer_arguments={"query_time": to_iso(parse_haystack_date(item["question_date"]))},
                    judge_arguments={
                        "query": item["question"],
                        "golden_answer": item["answer"],
                        "question_type": item["question_type"],
                    },
                ),
            ],
        )
        assert evaluation["success"], evaluation
        await case.export_evaluation(archive_path)
    finally:
        await case.close()

    names, files = _archive_members(archive_path.read_bytes())
    query_root = f"queries/{item['question_id']}"
    assert {"reme_workspace", "build_log", "queries"} <= names
    assert "build_log/build.log" in names
    assert f"{query_root}/answer.log" in names
    assert f"{query_root}/result.json" in names
    assert "queries/summary.json" in names
    summary = json.loads(files["queries/summary.json"])
    assert summary["queries"][0]["query_id"] == str(item["question_id"])
    assert "directory" not in summary["queries"][0]

    assert any(name.startswith("reme_workspace/daily/") and name.endswith(".md") for name in names)
    assert "reme_workspace/.git/HEAD" in names
    assert any(name.startswith("reme_workspace/.git/objects/") for name in names)
    assert any(name.startswith("reme_workspace/mem_session/") for name in names)
    assert any(name.startswith("reme_workspace/session/dialog/") and name.endswith(".jsonl") for name in names)
    excluded_prefixes = (
        "inbox",
        "tmp",
        "logs",
        "results",
    )
    assert not any(name == prefix or name.startswith(f"{prefix}/") for prefix in excluded_prefixes for name in names)
    assert not any(".reme" in PurePosixPath(name).parts for name in names)
