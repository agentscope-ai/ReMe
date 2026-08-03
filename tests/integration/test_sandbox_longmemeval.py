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
from sandbox import DockerReMeSandboxFactory, SourceCandidate, SourceSnapshot

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
async def test_first_longmemeval_case_analysis_export(tmp_path):
    """Ingest two sessions, answer once, and verify selective download paths."""
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
    archive_path = tmp_path / "analysis.tar.gz"
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

        for _, session_dt, session_id, messages in sessions:
            result = await case.ingest_session(
                session_id=session_id,
                messages=format_messages_for_reme(messages, session_dt),
                date=session_dt.strftime("%Y-%m-%d"),
            )
            assert result.success, result.error or result.answer

        answer = await case.answer(
            query=item["question"],
            query_time=to_iso(parse_haystack_date(item["question_date"])),
        )
        assert answer.success, answer.error or answer.answer
        await case.export(archive_path)
    finally:
        await case.close()

    names, files = _archive_members(archive_path.read_bytes())
    assert {"manifest.json", "runtime-layout.json", "logs", "results", "reme_workspace"} <= names
    assert "results/answer.json" in names
    assert "logs/actions.jsonl" in names
    assert any(name.startswith("logs/") and name.endswith(".log") for name in names)

    manifest = json.loads(files["manifest.json"])
    layout = json.loads(files["runtime-layout.json"])
    assert manifest["case_id"] == f"longmemeval-{item['question_id']}-smoke"
    assert manifest["export_profile"] == "analysis"
    assert layout["workspace_root"] == "reme_workspace"
    assert layout["configured_paths"]["daily_dir"] == "reme_workspace/daily"
    assert layout["configured_paths"]["mem_session_dir"] == "reme_workspace/mem_session"

    assert any(name.startswith("reme_workspace/daily/") and name.endswith(".md") for name in names)
    assert any(name.startswith("reme_workspace/mem_session/") for name in names)
    assert any(name.startswith("reme_workspace/session/dialog/") and name.endswith(".jsonl") for name in names)
    excluded_prefixes = (
        "inbox",
        "tmp",
        "reme_workspace/metadata",
        "reme_workspace/resource",
    )
    assert not any(name == prefix or name.startswith(f"{prefix}/") for prefix in excluded_prefixes for name in names)
    assert not any(".reme" in PurePosixPath(name).parts for name in names)
