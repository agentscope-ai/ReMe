"""Focused tests for Meta-ReMe validation orchestration."""

# Test doubles intentionally keep their methods compact.
# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

META_REME = Path(__file__).resolve().parents[2] / "meta-reme"
PROJECT_ROOT = META_REME.parent
for import_root in (META_REME, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

git_manager = importlib.import_module("git_manager")
models = importlib.import_module("models")
workspace_module = importlib.import_module("workspace")
adapters = importlib.import_module("validation.adapters")
evaluator = importlib.import_module("validation.evaluator")


def _domain(max_retries: int = 0):
    return models.DomainSpec(
        dataset=models.DatasetSpec(name="longmemeval", source="dataset.json", fingerprint="source-sha"),
        bundle_target="lme",
        benchmark_runner="benchmark.longmemeval.run",
        scorer="mean_query_score",
        scope=models.ScopeSpec(),
        sandbox=models.SandboxSpec(
            image="sandbox-base:test",
            timeout_seconds=60,
            concurrency=2,
            max_retries=max_retries,
        ),
        proposer=models.ProposerSpec(model="test-model"),
        budget=models.BudgetSpec(max_proposals=1),
    )


def _case(case_id: str):
    return models.CaseSpec(
        case_id=case_id,
        sessions=[
            models.SessionSpec(
                session_id=f"session-{case_id}",
                messages=[
                    {
                        "role": "user",
                        "content": f"remember {case_id}",
                        "created_at": "2026-08-07T08:00:00",
                    },
                ],
                metadata={"date": "2026-08-07"},
            ),
        ],
        queries=[
            models.QuerySpec(
                query_id=f"query-{case_id}",
                question="What should be remembered?",
                golden_answer=case_id,
                metadata={"question_type": "single-session-user"},
            ),
        ],
        metadata={"dataset": "longmemeval"},
    )


def _prepared_workspace(tmp_path: Path, case_ids: list[str]):
    workspace = workspace_module.Workspace.create(tmp_path / "workspace", _domain())
    for index, case_id in enumerate(case_ids):
        workspace.atomic_write_json(f"dataset/cases/{index:06d}.json", _case(case_id))
    workspace.atomic_write_json(
        "dataset/manifest.json",
        {
            "schema_version": 1,
            "dataset": "longmemeval",
            "source_fingerprint": "source-sha",
            "normalized_fingerprint": "normalized-sha",
            "case_count": len(case_ids),
            "query_count": len(case_ids),
        },
    )
    repository = workspace.path("code/repo/reme")
    (repository / "reme").mkdir(parents=True)
    (repository / "reme/__init__.py").write_text("VERSION = 'committed'\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0.0.0'\n", encoding="utf-8")
    commit = git_manager.initialize_repository(repository)
    return workspace, repository, commit


class _FakeFactory:
    def __init__(self, candidate, **kwargs):
        self.candidate = candidate
        self.kwargs = kwargs
        self.active = 0
        self.max_active = 0
        self.events = []

    async def create_case(self, case_id):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        return _FakeCase(self, case_id)


class _FakeCase:
    def __init__(self, factory, case_id):
        self.factory = factory
        self.case_id = case_id

    async def run_build(self, jobs):
        self.factory.events.append((self.case_id, "run_build"))
        assert [job.job for job in jobs] == ["auto_memory", "index_update"]
        await asyncio.sleep(0.01)
        return {"success": True, "jobs": []}

    async def run_queries(self, queries):
        self.factory.events.append((self.case_id, "run_queries"))
        await asyncio.sleep(0.01)
        return {
            "success": True,
            "summary": {"mean_score": 1.0, "query_count": len(queries)},
            "queries": [
                {
                    "query_id": query.query_id,
                    "question": query.question,
                    "golden_answer": query.golden_answer,
                    "answer": query.golden_answer,
                    "score": 1.0,
                    "error": None,
                }
                for query in queries
            ],
        }

    async def _export(self, destination, files):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, mode="w:gz") as archive:
            for name in files:
                payload = json.dumps({"case_id": self.case_id, "artifact": name}).encode()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        return destination

    async def export_workspace(self, destination):
        self.factory.events.append((self.case_id, "export_workspace"))
        return await self._export(destination, ["memory.json"])

    async def export_build_log(self, destination):
        self.factory.events.append((self.case_id, "export_build_log"))
        destination = Path(destination)
        destination.write_text("build log\n", encoding="utf-8")
        return destination

    async def export_queries(self, destination):
        self.factory.events.append((self.case_id, "export_queries"))
        query_root = f"queries/query-{self.case_id}"
        return await self._export(
            destination,
            ["queries/summary.json", f"{query_root}/answer.log", f"{query_root}/result.json"],
        )

    async def export_full(self, destination):
        return await self._export(destination, ["failure.json"])

    async def close(self):
        self.factory.active -= 1


def test_validation_uses_immutable_commit_and_bounds_full_case_lifecycle(tmp_path: Path) -> None:
    case_ids = ["case-1", "case-2", "case-3"]
    workspace, repository, commit = _prepared_workspace(tmp_path, case_ids)
    (repository / "reme/__init__.py").write_text("VERSION = 'dirty'\n", encoding="utf-8")
    created = []

    def build_factory(*args, **kwargs):
        factory = _FakeFactory(*args, **kwargs)
        created.append(factory)
        return factory

    subprocess.run(["git", "branch", "candidate-1", commit], cwd=repository, check=True)
    output = evaluator.run_validation(
        workspace.root,
        case_ids,
        "candidate-1",
        2,
        validation_id="validation-1",
        factory_builder=build_factory,
        environment={},
    )

    assert output == workspace.path("evaluations/candidate-1/validation-1")
    assert created[0].max_active == 2
    assert created[0].kwargs["config"] == "lme.yaml"
    with tarfile.open(fileobj=io.BytesIO(created[0].candidate.snapshot.archive), mode="r:gz") as archive:
        assert archive.extractfile("reme/__init__.py").read() == b"VERSION = 'committed'\n"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert summary["code_id"] == "candidate-1"
    assert summary["commit_sha"] == commit
    assert manifest["code_id"] == "candidate-1"
    assert manifest["commit_sha"] == commit
    assert [event for case_id, event in created[0].events if case_id == "case-1"] == [
        "run_build",
        "export_workspace",
        "export_build_log",
        "run_queries",
        "export_queries",
    ]
    assert summary["status"] == "completed"
    assert summary["mean_query_score"] == 1.0
    assert summary["case_count"] == 3
    attempt = output / "cases/case-1/attempt-1"
    assert (attempt / "memory_construction/result.json").is_file()
    assert (attempt / "memory_construction/build.log").is_file()
    memory_workspace = attempt / "memory_construction/reme_workspace.tar.gz"
    assert memory_workspace.is_file()
    with tarfile.open(memory_workspace, mode="r:gz") as archive:
        assert archive.getnames() == ["memory.json"]
    assert (attempt / "memory_construction/reme_workspace/memory.json").is_file()
    case_result = json.loads((attempt / "case_result.json").read_text(encoding="utf-8"))
    assert case_result["artifact_sha256"] == {
        "memory_workspace": evaluator._sha256_file(memory_workspace),  # pylint: disable=protected-access
    }
    assert (attempt / "queries/result.json").is_file()
    assert (attempt / "queries/query-case-1/answer.log").is_file()
    assert (attempt / "queries/query-case-1/result.json").is_file()
    assert not (attempt / "queries/summary.json").exists()
    assert not (attempt / "queries/artifacts").exists()
    assert {path.name for path in (attempt / "queries").iterdir()} == {"result.json", "query-case-1"}
    assert not list((attempt / "queries").glob("*.tar.gz"))
    assert not list((attempt / "queries").glob(".*.tar.gz"))


@pytest.mark.asyncio
async def test_validation_exposes_native_async_project_api(tmp_path: Path) -> None:
    workspace, _, _ = _prepared_workspace(tmp_path, ["case-async"])

    output = await evaluator.run_validation_async(
        workspace.root,
        ["case-async"],
        "init",
        1,
        validation_id="async-api",
        factory_builder=_FakeFactory,
        environment={},
    )

    assert output == workspace.path("evaluations/init/async-api")
    assert (output / "summary.json").is_file()


def test_validation_rejects_commit_sha_as_code_id(tmp_path: Path) -> None:
    workspace, _, commit = _prepared_workspace(tmp_path, ["case-1"])

    with pytest.raises(evaluator.ValidationError, match="unknown local branch code_id"):
        evaluator.run_validation(
            workspace.root,
            ["case-1"],
            commit,
            1,
            validation_id="commit-is-not-code-id",
            factory_builder=_FakeFactory,
            environment={},
        )

    with pytest.raises(evaluator.ValidationError, match="path-safe Git branch name"):
        evaluator.run_validation(
            workspace.root,
            ["case-1"],
            "feature/candidate",
            1,
            validation_id="unsafe-code-id",
            factory_builder=_FakeFactory,
            environment={},
        )


def test_adapters_build_dataset_specific_requests() -> None:
    lme_case = _case("lme")
    lme_case.queries[0].metadata["question_date"] = "2026/08/07 (Fri) 09:30"
    jobs = adapters.build_jobs(models.DatasetName.LONGMEMEVAL, lme_case)
    lme_query = adapters.evaluation_queries(models.DatasetName.LONGMEMEVAL, lme_case)[0]

    assert jobs[0].arguments["date"] == "2026-08-07"
    assert lme_query.answer_arguments == {"query_time": "2026-08-07T09:30:00"}
    assert lme_query.judge_answer_argument == "agent_answer"

    beam_case = models.CaseSpec(
        case_id="beam",
        sessions=lme_case.sessions,
        queries=[
            models.QuerySpec(
                query_id="event_ordering:1",
                question="What happened first?",
                golden_answer=["Mentions the first event"],
                metadata={"question_type": "event_ordering", "rubric": ["Mentions the first event"]},
            ),
        ],
    )
    beam_query = adapters.evaluation_queries(models.DatasetName.BEAM, beam_case)[0]
    assert beam_query.judge_answer_argument == "llm_response"
    assert beam_query.score_path == "metadata.llm_judge_score"
    assert beam_query.score_mapping is None


def test_longmemeval_build_filters_future_sessions() -> None:
    case = _case("lme-filter")
    case.queries[0].metadata["question_date"] = "2026/08/07 (Fri) 09:30"
    case.sessions.append(
        models.SessionSpec(
            session_id="future",
            messages=[{"role": "user", "content": "future", "created_at": "2026-08-08T08:00:00"}],
            metadata={"date": "2026/08/08 (Sat) 08:00"},
        ),
    )

    jobs = adapters.build_jobs(models.DatasetName.LONGMEMEVAL, case)

    assert [job.arguments.get("session_id") for job in jobs if job.job == "auto_memory"] == ["session-lme-filter"]
