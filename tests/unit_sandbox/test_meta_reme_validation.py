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
scheduler_module = importlib.import_module("validation.scheduler")


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


def _case(case_id: str, *, query_count: int = 1):
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
                query_id=f"query-{case_id}-{index}" if query_count > 1 else f"query-{case_id}",
                question="What should be remembered?",
                golden_answer=case_id,
                metadata={"question_type": "single-session-user"},
            )
            for index in range(query_count)
        ],
        metadata={"dataset": "longmemeval"},
    )


def _prepared_workspace(tmp_path: Path, case_ids: list[str], *, max_retries: int = 0):
    workspace = workspace_module.Workspace.create(tmp_path / "workspace", _domain(max_retries=max_retries))
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
        self.created = []
        self.resets = []

    async def create_case(self, case_id):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.created.append(case_id)
        return _FakeCase(self, case_id)


class _FakeCase:
    def __init__(self, factory, case_id):
        self.factory = factory
        self.case_id = case_id
        self.container_id = f"container-{len(factory.created)}"

    async def run_build(self, jobs):
        self.factory.events.append((self.case_id, "run_build"))
        assert [job.job for job in jobs] == ["auto_memory", "index_update"]
        assert jobs[0].memory_checkpoint is None
        assert jobs[1].memory_checkpoint == f"session: session-{self.case_id}"
        await asyncio.sleep(0.01)
        return {"success": True, "jobs": []}

    async def reset_case(self, case_id):
        self.factory.resets.append((self.case_id, case_id))
        self.case_id = case_id

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

    async def run_query(self, query):
        self.factory.events.append((self.case_id, f"run_query:{query.query_id}"))
        await asyncio.sleep(0.01)
        return {
            "query_id": query.query_id,
            "question": query.question,
            "golden_answer": query.golden_answer,
            "answer": query.golden_answer,
            "score": 1.0,
            "answer_result": {"success": True, "answer": query.golden_answer},
            "judge_result": {"success": True, "answer": "yes"},
            "error": None,
        }

    async def upload_workspace(self, source):
        assert Path(source).is_file()
        self.factory.events.append((self.case_id, "upload_workspace"))

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

    async def export_query(self, query_id, destination):
        self.factory.events.append((self.case_id, f"export_query:{query_id}"))
        return await self._export(
            destination,
            [f"queries/{query_id}/answer.log", f"queries/{query_id}/result.json"],
        )

    async def export_full(self, destination):
        return await self._export(destination, ["failure.json"])

    async def close(self):
        self.factory.active -= 1


def test_validation_uses_current_commit_and_bounds_full_case_lifecycle(tmp_path: Path) -> None:
    case_ids = ["case-1", "case-2", "case-3"]
    workspace, _, commit = _prepared_workspace(tmp_path, case_ids)
    created = []

    def build_factory(*args, **kwargs):
        factory = _FakeFactory(*args, **kwargs)
        created.append(factory)
        return factory

    output = evaluator.run_validation(
        workspace.root,
        case_ids,
        2,
        validation_id="validation-1",
        factory_builder=build_factory,
        environment={},
    )

    assert output == workspace.path(f"evaluations/init/{commit[:7]}/validation-1")
    assert created[0].max_active == 2
    assert len(created[0].created) == 2
    assert len(created[0].resets) == 2
    assert created[0].kwargs["config"] == "lme.yaml"
    with tarfile.open(fileobj=io.BytesIO(created[0].candidate.snapshot.archive), mode="r:gz") as archive:
        assert archive.extractfile("reme/__init__.py").read() == b"VERSION = 'committed'\n"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert summary["branch_name"] == "init"
    assert summary["commit_sha"] == commit
    assert manifest["branch_name"] == "init"
    assert manifest["commit_sha"] == commit
    assert manifest["container_reuse"] is True
    assert manifest["scheduling"] == {
        "construction_barrier": True,
        "query_unit": "single_query_lease",
        "workspace_affinity": "case_id",
    }
    assert [event for case_id, event in created[0].events if case_id == "case-1"] == [
        "run_build",
        "export_workspace",
        "export_build_log",
        "upload_workspace",
        "run_query:query-case-1",
        "export_query:query-case-1",
    ]
    assert summary["status"] == "completed"
    assert summary["mean_query_score"] == 1.0
    assert summary["case_count"] == 3
    case_output = output / "cases/case-1"
    assert (case_output / "memory_construction/result.json").is_file()
    assert (case_output / "memory_construction/build.log").is_file()
    memory_workspace = case_output / "memory_construction/reme_workspace.tar.gz"
    assert memory_workspace.is_file()
    with tarfile.open(memory_workspace, mode="r:gz") as archive:
        assert archive.getnames() == ["memory.json"]
    assert (case_output / "memory_construction/reme_workspace/memory.json").is_file()
    case_result = json.loads((case_output / "case_result.json").read_text(encoding="utf-8"))
    assert case_result["artifact_sha256"] == {
        "memory_workspace": evaluator._sha256_file(memory_workspace),  # pylint: disable=protected-access
    }
    assert (case_output / "queries/result.json").is_file()
    assert (case_output / "queries/query-case-1/answer.log").is_file()
    assert (case_output / "queries/query-case-1/result.json").is_file()
    assert not (case_output / "queries/summary.json").exists()
    assert not (case_output / "queries/artifacts").exists()
    assert {path.name for path in (case_output / "queries").iterdir()} == {"result.json", "query-case-1"}
    assert not list((case_output / "queries").glob("*.tar.gz"))
    assert not list((case_output / "queries").glob(".*.tar.gz"))


def test_validation_builds_every_memory_before_stealing_ordered_queries(tmp_path: Path) -> None:
    """Idle workers fan out one case only after the strict construction barrier."""

    workspace, _, _ = _prepared_workspace(tmp_path, ["case-many"])
    workspace.atomic_write_json("dataset/cases/000000.json", _case("case-many", query_count=6))
    created = []

    def build_factory(*args, **kwargs):
        factory = _FakeFactory(*args, **kwargs)
        created.append(factory)
        return factory

    output = evaluator.run_validation(
        workspace.root,
        ["case-many"],
        3,
        validation_id="query-stealing",
        factory_builder=build_factory,
        environment={},
    )

    factory = created[0]
    query_events = [event for _, event in factory.events if event.startswith("run_query:")]
    assert len(factory.created) == 3
    assert len(query_events) == 6
    assert [event for _, event in factory.events].index(query_events[0]) > max(
        index for index, (_, event) in enumerate(factory.events) if event == "export_build_log"
    )
    assert sum(event == "upload_workspace" for _, event in factory.events) == 2
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [query["query_id"] for query in summary["cases"][0]["queries"]] == [
        f"query-case-many-{index}" for index in range(6)
    ]


def test_validation_records_construction_infra_error_without_retrying(tmp_path: Path) -> None:
    """A failed construction is terminal while later cases continue on a fresh container."""

    class RetryCase(_FakeCase):
        """Fail the first build to emulate a broken container."""

        async def run_build(self, jobs):
            if self.factory.fail_next_build:
                self.factory.fail_next_build = False
                raise RuntimeError("container became unhealthy")
            return await super().run_build(jobs)

    class RetryFactory(_FakeFactory):
        """Create retry cases while tracking physical replacements."""

        def __init__(self, candidate, **kwargs):
            super().__init__(candidate, **kwargs)
            self.fail_next_build = True

        async def create_case(self, case_id):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.created.append(case_id)
            return RetryCase(self, case_id)

    workspace, _, _ = _prepared_workspace(tmp_path, ["case-1", "case-2"], max_retries=1)
    created = []

    def build_factory(*args, **kwargs):
        factory = RetryFactory(*args, **kwargs)
        created.append(factory)
        return factory

    output = evaluator.run_validation(
        workspace.root,
        ["case-1", "case-2"],
        1,
        validation_id="replace-infra-container",
        factory_builder=build_factory,
        environment={},
    )

    factory = created[0]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert factory.created == ["case-1", "case-2"]
    assert factory.resets == []
    assert factory.active == 0
    assert [result["status"] for result in summary["cases"]] == ["infra_error", "completed"]
    assert (output / "cases/case-1/failure.json").is_file()
    assert not list((output / "cases/case-1").glob("attempt-*"))


def test_validation_discards_a_container_when_reset_fails(tmp_path: Path) -> None:
    """A failed reset cannot leave the pool member available to later cases."""

    class ResetFailureCase(_FakeCase):
        """Reject assignment of the second logical case."""

        async def reset_case(self, case_id):
            if case_id == "case-2":
                raise RuntimeError("reset failed")
            await super().reset_case(case_id)

    class ResetFailureFactory(_FakeFactory):
        """Create reset-failure cases while tracking replacements."""

        async def create_case(self, case_id):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.created.append(case_id)
            return ResetFailureCase(self, case_id)

    case_ids = ["case-1", "case-2", "case-3"]
    workspace, _, _ = _prepared_workspace(tmp_path, case_ids)
    created = []

    def build_factory(*args, **kwargs):
        factory = ResetFailureFactory(*args, **kwargs)
        created.append(factory)
        return factory

    output = evaluator.run_validation(
        workspace.root,
        case_ids,
        1,
        validation_id="replace-reset-failure",
        factory_builder=build_factory,
        environment={},
    )

    factory = created[0]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert factory.created == ["case-1", "case-3"]
    assert factory.resets == [("case-3", "case-1")]
    assert factory.active == 0
    assert [result["status"] for result in summary["cases"]] == ["completed", "infra_error", "completed"]


def test_construction_candidate_failure_skips_case_queries(tmp_path: Path) -> None:
    """A structured build failure is terminal and never enters the query queue."""

    class FailedBuildCase(_FakeCase):
        """Return a structured candidate failure from construction."""

        async def run_build(self, jobs):
            self.factory.events.append((self.case_id, "run_build"))
            return {"success": False, "jobs": [{"job": jobs[0].job, "result": {"success": False}}]}

    class FailedBuildFactory(_FakeFactory):
        """Create cases whose candidate build fails normally."""

        async def create_case(self, case_id):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.created.append(case_id)
            return FailedBuildCase(self, case_id)

    workspace, _, _ = _prepared_workspace(tmp_path, ["case-1"])
    created = []

    def build_factory(*args, **kwargs):
        factory = FailedBuildFactory(*args, **kwargs)
        created.append(factory)
        return factory

    output = evaluator.run_validation(
        workspace.root,
        ["case-1"],
        2,
        validation_id="failed-construction",
        factory_builder=build_factory,
        environment={},
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["cases"][0]["status"] == "candidate_failure"
    assert not any(event.startswith("run_query:") for _, event in created[0].events)
    assert not (output / "cases/case-1/queries").exists()


def test_query_infrastructure_failure_is_saved_without_retrying(tmp_path: Path) -> None:
    """A failed query is terminal and its container is retired."""

    class FlakyQueryCase(_FakeCase):
        """Raise one infrastructure exception from the query primitive."""

        async def run_query(self, query):
            if self.factory.fail_next_query:
                self.factory.fail_next_query = False
                raise RuntimeError("query worker disconnected")
            return await super().run_query(query)

    class FlakyQueryFactory(_FakeFactory):
        """Create query cases while sharing the one-shot failure flag."""

        def __init__(self, candidate, **kwargs):
            super().__init__(candidate, **kwargs)
            self.fail_next_query = True

        async def create_case(self, case_id):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.created.append(case_id)
            return FlakyQueryCase(self, case_id)

    workspace, _, _ = _prepared_workspace(tmp_path, ["case-1"], max_retries=1)
    created = []

    def build_factory(*args, **kwargs):
        factory = FlakyQueryFactory(*args, **kwargs)
        created.append(factory)
        return factory

    output = evaluator.run_validation(
        workspace.root,
        ["case-1"],
        1,
        validation_id="retry-query",
        factory_builder=build_factory,
        environment={},
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    failure = output / "cases/case-1/queries/query-case-1/failure.json"
    assert summary["cases"][0]["status"] == "infra_error"
    assert summary["cases"][0]["queries"][0]["score"] is None
    assert created[0].created == ["case-1"]
    assert failure.is_file()
    assert json.loads(failure.read_text(encoding="utf-8"))["infrastructure_error"] is True


def test_query_failure_is_a_terminal_ordered_case_result(tmp_path: Path) -> None:
    """A failed lease cannot strand waiting workers or disappear from aggregation."""

    class BrokenQueryCase(_FakeCase):
        """Always lose the query worker response."""

        async def run_query(self, query):
            raise RuntimeError(f"lost {query.query_id}")

    class BrokenQueryFactory(_FakeFactory):
        """Create cases with a permanently broken query primitive."""

        async def create_case(self, case_id):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.created.append(case_id)
            return BrokenQueryCase(self, case_id)

    workspace, _, _ = _prepared_workspace(tmp_path, ["case-1"])
    output = evaluator.run_validation(
        workspace.root,
        ["case-1"],
        2,
        validation_id="exhaust-query",
        factory_builder=BrokenQueryFactory,
        environment={},
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    case_result = summary["cases"][0]
    assert summary["status"] == "infra_error"
    assert case_result["status"] == "infra_error"
    assert [query["query_id"] for query in case_result["queries"]] == ["query-case-1"]
    assert case_result["queries"][0]["infrastructure_error"] is True
    assert (output / "cases/case-1/failure.json").is_file()
    assert (output / "cases/case-1/queries/query-case-1/failure.json").is_file()


def test_fail_fast_construction_error_cancels_sibling_cases(tmp_path: Path) -> None:
    """The first terminal construction error cancels in-progress siblings."""

    class FailFastBuildCase(_FakeCase):
        """Fail one build after allowing its sibling to start."""

        async def run_build(self, jobs):
            if self.case_id == "case-fail":
                await asyncio.sleep(0.02)
                return {"success": False, "jobs": []}
            self.factory.sibling_started = True
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.factory.sibling_cancelled = True
                raise
            return await super().run_build(jobs)

    class FailFastBuildFactory(_FakeFactory):
        """Track cancellation of the non-failing construction sandbox."""

        def __init__(self, candidate, **kwargs):
            super().__init__(candidate, **kwargs)
            self.sibling_started = False
            self.sibling_cancelled = False

        async def create_case(self, case_id):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.created.append(case_id)
            return FailFastBuildCase(self, case_id)

    workspace, _, commit = _prepared_workspace(tmp_path, ["case-fail", "case-slow"])
    created = []

    def build_factory(*args, **kwargs):
        factory = FailFastBuildFactory(*args, **kwargs)
        created.append(factory)
        return factory

    with pytest.raises(evaluator.ValidationFailFastError, match="case-fail"):
        evaluator.run_validation(
            workspace.root,
            ["case-fail", "case-slow"],
            2,
            validation_id="fail-fast-build",
            fail_fast=True,
            factory_builder=build_factory,
            environment={},
        )

    output = workspace.path(f"evaluations/init/{commit[:7]}/fail-fast-build")
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert created[0].sibling_started is True
    assert created[0].sibling_cancelled is True
    assert created[0].active == 0
    assert failure["status"] == "aborted"
    assert failure["fail_fast"] is True
    assert not (output / "summary.json").exists()


def test_fail_fast_query_error_does_not_run_later_queries(tmp_path: Path) -> None:
    """A structured answer/judge error stops later leases without reclassifying it as infrastructure."""

    class FailedQueryCase(_FakeCase):
        """Return one normal structured query failure."""

        async def run_query(self, query):
            self.factory.events.append((self.case_id, f"run_query:{query.query_id}"))
            return {
                "query_id": query.query_id,
                "question": query.question,
                "golden_answer": query.golden_answer,
                "answer": None,
                "score": None,
                "answer_result": {"success": False},
                "judge_result": None,
                "error": "answer job failed",
            }

    class FailedQueryFactory(_FakeFactory):
        """Create cases whose first query returns a structured error."""

        async def create_case(self, case_id):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.created.append(case_id)
            return FailedQueryCase(self, case_id)

    workspace, _, commit = _prepared_workspace(tmp_path, ["case-1"])
    workspace.atomic_write_json("dataset/cases/000000.json", _case("case-1", query_count=2))
    created = []

    def build_factory(*args, **kwargs):
        factory = FailedQueryFactory(*args, **kwargs)
        created.append(factory)
        return factory

    with pytest.raises(evaluator.ValidationFailFastError, match="answer job failed"):
        evaluator.run_validation(
            workspace.root,
            ["case-1"],
            1,
            validation_id="fail-fast-query",
            fail_fast=True,
            factory_builder=build_factory,
            environment={},
        )

    output = workspace.path(f"evaluations/init/{commit[:7]}/fail-fast-query")
    assert [event for _, event in created[0].events if event.startswith("run_query:")] == [
        "run_query:query-case-1-0",
    ]
    assert (output / "cases/case-1/queries/query-case-1-0/result.json").is_file()
    assert not (output / "cases/case-1/queries/query-case-1-1").exists()
    assert (output / "failure.json").is_file()
    assert not (output / "summary.json").exists()


@pytest.mark.asyncio
async def test_validation_exposes_native_async_project_api(tmp_path: Path) -> None:
    workspace, _, commit = _prepared_workspace(tmp_path, ["case-async"])

    output = await evaluator.run_validation_async(
        workspace.root,
        ["case-async"],
        1,
        validation_id="async-api",
        factory_builder=_FakeFactory,
        environment={},
    )

    assert output == workspace.path(f"evaluations/init/{commit[:7]}/async-api")
    assert (output / "summary.json").is_file()


@pytest.mark.parametrize("dirty_kind", ["tracked", "staged", "untracked"])
def test_validation_rejects_uncommitted_code(tmp_path: Path, dirty_kind: str) -> None:
    workspace, repository, commit = _prepared_workspace(tmp_path, ["case-1"])
    if dirty_kind == "untracked":
        (repository / "untracked.py").write_text("DIRTY = True\n", encoding="utf-8")
    else:
        (repository / "reme/__init__.py").write_text("VERSION = 'dirty'\n", encoding="utf-8")
        if dirty_kind == "staged":
            subprocess.run(["git", "add", "reme/__init__.py"], cwd=repository, check=True)

    with pytest.raises(evaluator.ValidationError, match="uncommitted changes"):
        evaluator.run_validation(
            workspace.root,
            ["case-1"],
            1,
            validation_id="dirty-code",
            factory_builder=_FakeFactory,
            environment={},
        )
    assert not workspace.path(f"evaluations/init/{commit[:7]}/dirty-code").exists()


def test_validation_rejects_detached_head_and_non_path_safe_branch(tmp_path: Path) -> None:
    workspace, repository, commit = _prepared_workspace(tmp_path, ["case-1"])
    subprocess.run(["git", "checkout", "--detach", commit], cwd=repository, check=True, capture_output=True)
    with pytest.raises(evaluator.ValidationError, match="detached HEAD"):
        evaluator.run_validation(workspace.root, ["case-1"], 1, factory_builder=_FakeFactory, environment={})

    subprocess.run(["git", "switch", "-c", "feature/candidate"], cwd=repository, check=True, capture_output=True)
    with pytest.raises(evaluator.ValidationError, match="path-safe"):
        evaluator.run_validation(workspace.root, ["case-1"], 1, factory_builder=_FakeFactory, environment={})


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


@pytest.mark.asyncio
async def test_query_scheduler_prefers_loaded_memory_and_terminally_records_failures() -> None:
    first = scheduler_module.QueryCasePlan(0, "case-1", ("q1", "q2"))
    second = scheduler_module.QueryCasePlan(1, "case-2", ("q3",))
    scheduler = scheduler_module.QueryScheduler([first, second])

    owner = await scheduler.claim(0, first.case_id)
    thief = await scheduler.claim(1, first.case_id)
    assert owner.plan is first
    assert owner.selection == "affinity"
    assert thief.plan is first
    assert thief.selection == "affinity_steal"

    await scheduler.fail(owner, {"query_id": "q1", "error": "infra"})
    with pytest.raises(RuntimeError, match="stale"):
        await scheduler.complete(owner, {"query_id": "q1"})

    await scheduler.complete(thief, {"query_id": "q2", "score": 1.0})
    remaining = await scheduler.claim(1, first.case_id)
    assert remaining.plan is second
    await scheduler.complete(remaining, {"query_id": "q3", "score": 1.0})
    assert await scheduler.claim(0, None) is None


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
