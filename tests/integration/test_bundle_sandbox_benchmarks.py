"""End-to-end bundle-builder coverage against the first benchmark cases.

This is an opt-in integration test: it requires the benchmark datasets, a
running Docker daemon, the ReMe sandbox base image, and real model credentials.
The test intentionally invokes the builder through its CLI before constructing
either ``SourceCandidate`` so it covers the same artifacts users generate.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterable

from dotenv import load_dotenv
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
from sandbox import DockerReMeSandboxFactory, SourceCandidate, SourceSnapshot  # noqa: E402

BUNDLE_BUILDER = PROJECT_ROOT / "meta-reme/bundle_builder.py"
LME_DATASET = PROJECT_ROOT / "benchmark/datasets/longmemeval/longmemeval_s_reme_cleaned.json"
DEFAULT_BEAM_DATASET = PROJECT_ROOT / "benchmark/datasets/BEAM"
SANDBOX_ENV_NAMES = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_BACKEND",
    "LLM_MODEL_NAME",
    "BENCH_MODEL_NAME",
    "JUDGE_MODEL_NAME",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_BACKEND",
    "EMBEDDING_MODEL_NAME",
)


def _load_benchmark_module(name: str, relative_path: str) -> ModuleType:
    """Load a runner without requiring ``benchmark`` to be a Python package."""
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BEAM_RUNNER = _load_benchmark_module("reme_beam_runner", "benchmark/beam/run.py")
LME_RUNNER = _load_benchmark_module("reme_lme_runner", "benchmark/longmemeval/run.py")


def _sandbox_environment() -> dict[str, str]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    missing = [name for name in ("LLM_API_KEY", "EMBEDDING_API_KEY") if not os.environ.get(name)]
    if missing:
        pytest.skip(f"missing sandbox integration credentials: {', '.join(missing)}")
    return {name: os.environ[name] for name in SANDBOX_ENV_NAMES if os.environ.get(name)}


def _beam_dataset_root() -> Path:
    configured = os.environ.get("BEAM_DATASET_ROOT")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_BEAM_DATASET


def _load_beam_questions(path: Path) -> list[tuple[str, str, list[str]]]:
    """Flatten BEAM's question-type mapping while preserving dataset order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"BEAM probing questions must be an object: {path}")
    questions: list[tuple[str, str, list[str]]] = []
    for question_type, entries in payload.items():
        assert isinstance(entries, list), f"BEAM question group {question_type!r} must be a list"
        for entry in entries:
            assert isinstance(entry, dict) and isinstance(entry.get("question"), str)
            rubric = entry.get("rubric", [])
            assert isinstance(rubric, list) and all(isinstance(value, str) for value in rubric)
            questions.append((str(question_type), entry["question"], rubric))
    return questions


async def _ingest_sessions(case: Any, sessions: Iterable[dict[str, Any]]) -> int:
    count = 0
    for session in sessions:
        result = await case.ingest_session(
            session_id=session["session_id"],
            messages=session["messages"],
            date=session["date"],
        )
        assert result.success, result.error or result.answer
        await case.commit_memory_history(f"session: {session['session_id']}")
        count += 1
    return count


async def _run_first_lme_case(bundle_root: Path, env: dict[str, str], artifact_dir: Path) -> None:
    dataset = json.loads(LME_DATASET.read_text(encoding="utf-8"))
    assert isinstance(dataset, list) and dataset, f"LongMemEval dataset is empty: {LME_DATASET}"
    item = dataset[0]
    question_time = LME_RUNNER.parse_haystack_date(item["question_date"])
    sessions = [
        {
            "session_id": session_id,
            "messages": LME_RUNNER.format_messages_for_reme(messages, session_time),
            "date": session_time.strftime("%Y-%m-%d"),
        }
        for _, session_time, session_id, messages in LME_RUNNER.sessions_sorted_by_time(item)
        if session_time <= question_time
    ]

    candidate = SourceCandidate(SourceSnapshot.from_directory(bundle_root))
    factory = DockerReMeSandboxFactory(candidate, env=env, config="lme.yaml", command_timeout=1800.0)
    case = await factory.create_case(f"bundle-lme-{item['question_id']}")
    try:
        assert await _ingest_sessions(case, sessions) == len(sessions)
        digest = await case.run_job("digest_update")
        assert digest.success, digest.error or digest.answer
        answer = await case.answer(query=item["question"], query_time=LME_RUNNER.to_iso(question_time))
        assert answer.success and answer.answer, answer.error or answer.answer
        judgment = await case.judge(
            query=item["question"],
            agent_answer=answer.answer,
            golden_answer=item["answer"],
            question_type=item["question_type"],
        )
        assert judgment.success, judgment.error or judgment.answer
        await case.export(artifact_dir / "lme-first-case.tar.gz")
    finally:
        await case.close()


async def _run_first_beam_100k_case(
    bundle_root: Path,
    beam_root: Path,
    env: dict[str, str],
    artifact_dir: Path,
) -> None:
    case_ids = BEAM_RUNNER.get_available_cases(beam_root, "100K")
    assert case_ids, f"BEAM 100K contains no cases: {beam_root}"
    case_id = case_ids[0]
    case_dir = beam_root / "chats" / "100K" / case_id
    sessions = BEAM_RUNNER.load_beam_chat(case_dir / "chat.json", "100K", case_id)
    questions = _load_beam_questions(case_dir / "probing_questions/probing_questions.json")
    assert sessions and questions

    candidate = SourceCandidate(SourceSnapshot.from_directory(bundle_root))
    factory = DockerReMeSandboxFactory(candidate, env=env, config="beam.yaml", command_timeout=1800.0)
    case = await factory.create_case(f"bundle-beam-100k-{case_id}")
    try:
        assert await _ingest_sessions(case, sessions) == len(sessions)
        digest = await case.run_job("digest_update")
        assert digest.success, digest.error or digest.answer
        for question_type, question, rubric in questions:
            answer = await case.answer(query=question)
            assert answer.success and answer.answer, answer.error or answer.answer
            judgment = await case.run_job(
                "answer_judge",
                {
                    "llm_response": answer.answer,
                    "rubric": rubric,
                    "probing_question": question,
                    "question_type": question_type,
                },
            )
            assert judgment.success, judgment.error or judgment.answer
        await case.export(artifact_dir / "beam-100k-first-case.tar.gz")
    finally:
        await case.close()


@pytest.mark.asyncio
async def test_generated_bundles_run_first_lme_and_beam_100k_cases(tmp_path: Path) -> None:
    """Build via CLI, then evaluate the first complete case of both datasets."""
    output_dir = tmp_path / "bundles"
    subprocess.run(
        [sys.executable, str(BUNDLE_BUILDER), "--output-dir", str(output_dir)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    lme_bundle = output_dir / "lme/reme"
    beam_bundle = output_dir / "beam/reme"
    assert lme_bundle.is_dir() and beam_bundle.is_dir()

    env = _sandbox_environment()
    if not LME_DATASET.is_file():
        pytest.skip(f"LongMemEval dataset is unavailable: {LME_DATASET}")
    beam_root = _beam_dataset_root()
    if not (beam_root / "chats/100K").is_dir():
        pytest.skip(f"BEAM 100K dataset is unavailable: {beam_root}")

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    await _run_first_lme_case(lme_bundle, env, artifact_dir)
    await _run_first_beam_100k_case(beam_bundle, beam_root, env, artifact_dir)
    assert (artifact_dir / "lme-first-case.tar.gz").stat().st_size > 0
    assert (artifact_dir / "beam-100k-first-case.tar.gz").stat().st_size > 0
