"""Pool ordering, reuse, and cleanup without Docker or model calls."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access,unused-argument

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from deploy.benchmark_docker import (
    BenchmarkDockerPool,
    CasePlan,
    DockerReMeSandboxFactory,
    EvaluationQuery,
    JobRequest,
    SourceCandidate,
    SourceSnapshot,
)
from deploy.benchmark_docker.docker_sandbox import SandboxCommandError, SandboxCommandTimeoutError, _archive_workspace
from deploy.benchmark_docker import worker as sandbox_worker


class FakeCase:
    def __init__(self, name: str, events: list[tuple]) -> None:
        self.name = name
        self.container_id = name
        self.events = events
        self.case_id = name
        self.reusable = True

    async def reset_case(self, case_id: str) -> None:
        self.case_id = case_id
        self.events.append((case_id, "reset"))

    async def upload_workspace(self, source: Path) -> None:
        assert source.exists()
        self.events.append((self.case_id, "upload", source.read_text(encoding="utf-8")))

    async def run_build(self, jobs: list[JobRequest]) -> dict:
        self.events.append((self.case_id, "build", len(jobs)))
        await asyncio.sleep(0)
        return {"success": True}

    async def export_workspace(self, destination: Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.case_id, encoding="utf-8")
        self.events.append((self.case_id, "export_workspace"))
        return destination

    async def run_queries(self, queries: list[EvaluationQuery]) -> dict:
        self.events.append((self.case_id, "query", len(queries)))
        return {"success": True}

    async def export_evaluation(self, destination: Path) -> Path:
        self.events.append((self.case_id, "export_evaluation"))
        return destination

    async def export_build_log(self, destination: Path) -> Path:
        destination = Path(destination)
        destination.write_text("build failed", encoding="utf-8")
        self.events.append((self.case_id, "export_build_log"))
        return destination

    async def read_build_log(self) -> str:
        self.events.append((self.case_id, "read_build_log"))
        return "build failed"

    async def close(self) -> None:
        self.events.append((self.name, "close"))


class FakeFactory:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    async def create_cases(self, case_ids: list[str], *, concurrency: int) -> list[FakeCase]:
        assert len(case_ids) <= concurrency
        return [FakeCase(name, self.events) for name in case_ids]


@pytest.mark.asyncio
async def test_pool_builds_and_queries_each_case_then_reuses_workspace(tmp_path):
    events: list[tuple] = []
    pool = BenchmarkDockerPool(FakeFactory(events), concurrency=1)
    # Put the dependent first to exercise task scheduling independently of input order.
    plans = [
        CasePlan(
            "child",
            queries=(EvaluationQuery("q2", "question", "secret"),),
            reuse_workspace_from="parent",
            export_evaluation_to=tmp_path / "evaluation.tar.gz",
        ),
        CasePlan("parent", build_jobs=(JobRequest("build"),), queries=(EvaluationQuery("q1", "question", "secret"),)),
    ]
    results = await pool.run(plans)

    assert [result.case_id for result in results] == ["child", "parent"]
    assert all(result.success for result in results)
    assert ("child", "upload", "parent") in events
    assert ("child", "export_evaluation") in events
    assert events.index(("parent", "build", 1)) < events.index(("parent", "query", 1))
    assert events.index(("parent", "export_workspace")) < events.index(("parent", "query", 1))
    assert events.index(("parent", "query", 1)) < events.index(("child", "query", 1))
    assert events[-1] == ("pool-0", "close")
    assert all(result.workspace_snapshot is None for result in results)


@pytest.mark.asyncio
async def test_pool_rejects_workspace_cycle_before_creating_containers():
    events: list[tuple] = []
    pool = BenchmarkDockerPool(FakeFactory(events), concurrency=2)
    with pytest.raises(ValueError, match="cycle"):
        await pool.run([CasePlan("a", reuse_workspace_from="b"), CasePlan("b", reuse_workspace_from="a")])
    assert not events


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True], ids=["failed-result", "exception"])
async def test_pool_keeps_build_log_on_failure(tmp_path, raises):
    class FailedBuildCase(FakeCase):
        async def run_build(self, jobs):
            if raises:
                raise TimeoutError("build exceeded command timeout")
            return {"success": False, "jobs": [{"error": "failed"}]}

    class FailedBuildFactory(FakeFactory):
        async def create_cases(self, case_ids, *, concurrency):
            return [FailedBuildCase(name, self.events) for name in case_ids]

    events: list[tuple] = []
    log_path = tmp_path / "build.log"
    plan = CasePlan("failed", build_jobs=(JobRequest("build"),), export_build_log_to=log_path)

    result = (await BenchmarkDockerPool(FailedBuildFactory(events), concurrency=1).run([plan]))[0]

    assert not result.success
    if raises:
        assert result.error == "TimeoutError: build exceeded command timeout"
        assert result.build is None
    else:
        assert result.error == "build failed"
        assert result.build == {"success": False, "jobs": [{"error": "failed"}]}
    assert result.build_log == "build failed"
    assert log_path.read_text(encoding="utf-8") == "build failed"
    assert events.index(("failed", "read_build_log")) < events.index(("pool-0", "close"))


@pytest.mark.asyncio
async def test_pool_replacement_failure_preserves_healthy_cases():
    events: list[tuple] = []

    class Worker(FakeCase):
        async def run_build(self, jobs):
            if self.case_id == "broken":
                self.reusable = False
                raise RuntimeError("command failed")
            await asyncio.sleep(0.02)
            self.events.append((self.case_id, "finished"))
            return {"success": True}

    class Factory(FakeFactory):
        async def create_cases(self, case_ids, *, concurrency):
            return [Worker(name, self.events) for name in case_ids]

        async def create_case(self, case_id):
            raise RuntimeError("replacement failed")

    plans = [CasePlan(case_id, build_jobs=(JobRequest("build"),)) for case_id in ("broken", "healthy", "waiting")]
    results = await asyncio.wait_for(BenchmarkDockerPool(Factory(events), concurrency=2).run(plans), timeout=2)

    assert [result.case_id for result in results] == ["broken", "healthy", "waiting"]
    assert "command failed" in results[0].error
    assert "replacement failed" in results[0].error
    assert results[1].success and results[2].success
    assert ("healthy", "finished") in events and ("waiting", "finished") in events


@pytest.mark.asyncio
async def test_pool_exhaustion_unblocks_waiting_and_dependent_cases():
    events: list[tuple] = []

    class BrokenCase(FakeCase):
        async def run_build(self, jobs):
            self.reusable = False
            raise RuntimeError("command failed")

    class Factory(FakeFactory):
        async def create_cases(self, case_ids, *, concurrency):
            return [BrokenCase(name, self.events) for name in case_ids]

        async def create_case(self, case_id):
            raise RuntimeError("replacement failed")

    plans = [
        CasePlan("dependent", reuse_workspace_from="waiting"),
        CasePlan("broken", build_jobs=(JobRequest("build"),)),
        CasePlan("waiting", build_jobs=(JobRequest("build"),)),
        CasePlan("waiting-2", build_jobs=(JobRequest("build"),)),
    ]
    results = await asyncio.wait_for(BenchmarkDockerPool(Factory(events), concurrency=1).run(plans), timeout=2)

    assert results[0].error == "source case 'waiting' failed"
    assert "replacement failed" in results[1].error
    assert all("no available workers" in result.error for result in results[2:])
    assert not any(case_id.startswith("waiting") for case_id, action in events if action == "reset")


@pytest.mark.asyncio
async def test_pool_close_failure_retires_slot_without_replacement():
    events: list[tuple] = []

    class BrokenCase(FakeCase):
        async def run_build(self, jobs):
            self.reusable = False
            raise RuntimeError("command failed")

        async def close(self):
            self.events.append((self.name, "close failed"))
            raise RuntimeError("close failed")

    class Factory(FakeFactory):
        async def create_cases(self, case_ids, *, concurrency):
            return [BrokenCase(name, self.events) for name in case_ids]

        async def create_case(self, case_id):
            pytest.fail("replacement must not start while the old container may still be running")

    plans = [CasePlan(case_id, build_jobs=(JobRequest("build"),)) for case_id in ("broken", "next")]
    results = await asyncio.wait_for(BenchmarkDockerPool(Factory(events), concurrency=1).run(plans), timeout=2)

    assert "close failed" in results[0].error
    assert "no available workers" in results[1].error
    assert "close failed" in results[1].error
    assert ("next", "build", 1) not in events


@pytest.mark.asyncio
async def test_pool_cancels_case_before_closing_container():
    started = asyncio.Event()
    events: list[tuple] = []

    class BlockingCase(FakeCase):
        async def run_queries(self, queries):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.events.append((self.case_id, "query_stopped"))

    class BlockingFactory(FakeFactory):
        async def create_cases(self, case_ids, *, concurrency):
            return [BlockingCase(name, self.events) for name in case_ids]

    plan = CasePlan("case", workspace_source=Path(__file__), queries=(EvaluationQuery("q", "question", "secret"),))
    task = asyncio.create_task(BenchmarkDockerPool(BlockingFactory(events), concurrency=1).run([plan]))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events.index(("case", "query_stopped")) < events.index(("pool-0", "close"))


@pytest.mark.asyncio
async def test_pool_final_close_failure_marks_only_cases_using_that_worker():
    events: list[tuple] = []

    class Worker(FakeCase):
        async def run_build(self, jobs):
            if self.case_id == "failed-build":
                return {"success": False}
            return await super().run_build(jobs)

        async def close(self):
            if self.name == "pool-0":
                raise RuntimeError("delete failed")
            await super().close()

    class Factory(FakeFactory):
        async def create_cases(self, case_ids, *, concurrency):
            return [Worker(name, self.events) for name in case_ids]

    plans = [
        CasePlan(case_id, build_jobs=(JobRequest("build"),)) for case_id in ("first", "healthy", "failed-build", "last")
    ]
    results = await BenchmarkDockerPool(Factory(events), concurrency=2).run(plans)
    assert not results[0].success
    assert "pool-0" in results[0].error and "delete failed" in results[0].error
    assert results[1].success
    assert results[2].error.startswith("build failed;")
    assert "delete failed" in results[2].error
    assert results[3].success


@pytest.mark.asyncio
async def test_pool_reports_unused_worker_cleanup_failure():
    events: list[tuple] = []

    class Worker(FakeCase):
        async def run_build(self, jobs):
            return {"success": False}

        async def close(self):
            if self.name == "pool-1":
                raise RuntimeError("unused worker leaked")

    class Factory(FakeFactory):
        async def create_cases(self, case_ids, *, concurrency):
            return [Worker(name, self.events) for name in case_ids]

    plans = [CasePlan("parent", build_jobs=(JobRequest("build"),)), CasePlan("child", reuse_workspace_from="parent")]
    results = await BenchmarkDockerPool(Factory(events), concurrency=2).run(plans)
    assert results[0].error.startswith("build failed;")
    assert results[1].error.startswith("source case 'parent' failed;")
    assert all("unused worker leaked" in result.error for result in results)


@pytest.mark.asyncio
async def test_pool_finishes_final_cleanup_despite_repeated_cancellation():
    started, release = asyncio.Event(), asyncio.Event()
    events: list[tuple] = []

    class Worker(FakeCase):
        async def close(self):
            started.set()
            await release.wait()
            self.events.append((self.name, "closed"))
            raise RuntimeError("delete failed")

    class Factory(FakeFactory):
        async def create_cases(self, case_ids, *, concurrency):
            return [Worker(name, self.events) for name in case_ids]

    task = asyncio.create_task(
        BenchmarkDockerPool(Factory(events), concurrency=1).run([CasePlan("case", build_jobs=(JobRequest("build"),))]),
    )
    await asyncio.wait_for(started.wait(), timeout=2)
    for _ in range(2):
        task.cancel()
        await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError) as failure:
        await asyncio.wait_for(task, timeout=2)
    assert events[-1] == ("pool-0", "closed")
    assert any("delete failed" in note for note in failure.value.__notes__)


@pytest.mark.asyncio
async def test_pool_does_not_replace_worker_for_cases_already_skipped():
    events: list[tuple] = []

    class Worker(FakeCase):
        async def run_build(self, jobs):
            if self.case_id == "parent":
                return {"success": False}
            # Let the dependent observe the failed parent before retirement.
            await asyncio.sleep(0)
            self.reusable = False
            raise RuntimeError("command timed out")

    class Factory(FakeFactory):
        async def create_cases(self, case_ids, *, concurrency):
            return [Worker(name, self.events) for name in case_ids]

        async def create_case(self, case_id):
            self.events.append((case_id, "replacement"))
            return FakeCase(case_id, self.events)

    plans = [
        CasePlan("parent", build_jobs=(JobRequest("build"),)),
        CasePlan("skipped", reuse_workspace_from="parent"),
        CasePlan("last", build_jobs=(JobRequest("build"),)),
    ]
    results = await BenchmarkDockerPool(Factory(events), concurrency=1).run(plans)
    assert results[1].error == "source case 'parent' failed"
    assert "command timed out" in results[2].error
    assert not any(action == "replacement" for _, action in events)


def test_source_snapshot_prunes_excluded_directories_and_secret_files(tmp_path, monkeypatch):
    (tmp_path / "reme").mkdir()
    (tmp_path / "reme" / "__init__.py").write_text("ok", encoding="utf-8")
    excluded = [tmp_path / name for name in ("baselines", ".venv", "node_modules")]
    for directory in excluded:
        (directory / "nested").mkdir(parents=True)
        (directory / "nested" / "large.json").write_text("private", encoding="utf-8")
    (tmp_path / ".env").write_text("secret", encoding="utf-8")
    scandir = os.scandir

    def checked_scandir(path):
        path = Path(path)
        assert not any(path == directory or directory in path.parents for directory in excluded)
        return scandir(path)

    monkeypatch.setattr(os, "scandir", checked_scandir)
    snapshot = SourceSnapshot.from_directory(tmp_path)
    assert snapshot.file_count == 1


def test_source_snapshot_preserves_custom_path_rules_and_symlink_checks(tmp_path):
    for name in ("pkg/cache/keep.py", "pkg/core.py", "pkg/cache/private.secret"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    (tmp_path / "pkg" / "ignored").symlink_to(tmp_path / "pkg" / "cache", target_is_directory=True)
    patterns = ("pkg/cache", "pkg/ignored", "pkg/*.secret")
    snapshot = SourceSnapshot.from_directory(tmp_path, excludes=patterns)
    with tarfile.open(fileobj=io.BytesIO(snapshot.archive), mode="r:gz") as archive:
        assert archive.getnames() == ["pkg/cache/keep.py", "pkg/core.py"]
    (tmp_path / "link").symlink_to(tmp_path / "pkg" / "cache", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link: link"):
        SourceSnapshot.from_directory(tmp_path, excludes=patterns)


def test_worker_merges_plugin_defaults_before_validating_config(tmp_path, monkeypatch):
    from reme.plugin import Plugin, PluginManager

    defaults = {
        "daily_dir": "notes",
        "digest_dir": "summaries",
        "jobs": {"plugin_version": {"backend": "base", "steps": [{"backend": "version_step"}]}},
    }
    manager = PluginManager([Plugin(name="example", config=defaults)])
    monkeypatch.setattr(PluginManager, "discover", classmethod(lambda cls, specs: manager))
    # _prepare_runtime changes cwd and TMPDIR in the disposable worker process.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "digest_dir: custom_digest\njobs:\n  plugin_version:\n    enable_serve: false\n",
        encoding="utf-8",
    )
    case_root, prepared = sandbox_worker._prepare_runtime(
        {
            "case_root": str(tmp_path),
            "workspace_dir": str(tmp_path / "reme_workspace"),
            "config": str(config),
            "plugins": ["example"],
        },
    )

    assert prepared["daily_dir"] == "notes"
    assert prepared["digest_dir"] == "custom_digest"
    job = prepared["jobs"]["plugin_version"]
    assert job["backend"] == "base"
    assert job["steps"] == [{"backend": "version_step"}]
    assert job["enable_serve"] is False
    layout = json.loads((case_root / "runtime-layout.json").read_text(encoding="utf-8"))
    assert layout["configured_paths"]["daily_dir"] == "reme_workspace/notes"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["job", "queries"])
async def test_worker_keeps_startup_error_or_cancellation_when_close_fails(tmp_path, monkeypatch, mode):
    import reme.reme as reme_module

    primary = RuntimeError("startup exploded") if mode == "job" else asyncio.CancelledError("startup cancelled")
    closed = False

    class FakeReMe:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            raise primary

        async def close(self):
            nonlocal closed
            closed = True
            raise RuntimeError("close exploded")

    monkeypatch.setattr(reme_module, "ReMe", FakeReMe)
    monkeypatch.setattr(sandbox_worker, "_prepare_runtime", lambda _request: (tmp_path, {}))
    runner = sandbox_worker._run if mode == "job" else sandbox_worker._run_queries
    with pytest.raises(type(primary)) as failure:
        await runner({"job": "build", "queries": [{"query_id": "q1", "question": "Question?"}]})
    assert failure.value is primary
    assert closed
    assert any("close exploded" in note for note in primary.__notes__)


@pytest.mark.parametrize("mode", ["job", "build", "queries"])
@pytest.mark.parametrize("job_fails", [True, "result", False], ids=["exception", "failed-result", "close-only"])
def test_worker_response_preserves_failed_job_when_close_fails(tmp_path, monkeypatch, mode, job_fails):
    import reme.reme as reme_module
    from types import SimpleNamespace
    from reme.utils.counter import global_counter_inc

    class FakeReMe:
        def __init__(self, **_kwargs):
            self.context = SimpleNamespace(components={}, jobs={"build": object()}, metadata={})

        async def start(self):
            pass

        async def run_job(self, job, **_kwargs):
            global_counter_inc(self.context.metadata, ["__job_counter", job])
            if job_fails == "result":
                return SimpleNamespace(success=False, answer="job failed", metadata={"diagnostic": "original"})
            if job_fails:
                raise RuntimeError("job exploded")
            return SimpleNamespace(success=True, answer="done", metadata={})

        async def close(self):
            raise ValueError("close exploded")

    monkeypatch.setattr(reme_module, "ReMe", FakeReMe)
    monkeypatch.setattr(sandbox_worker, "_prepare_runtime", lambda _request: (tmp_path, {}))
    request = {
        "mode": mode,
        "case_root": str(tmp_path),
        "job": "build",
        "jobs": [{"job": "build"}],
        "queries": [{"query_id": "q1", "question": "Question?", "answer_job": "build"}],
    }
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["worker", "--request", str(request_path), "--response", str(response_path)])
    assert sandbox_worker.main() == 1
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["success"] is False
    if not job_fails:
        assert response["error"] == "ValueError: close exploded"
        return

    result = response
    if mode == "build":
        result = response["jobs"][0]["result"]
        log = (tmp_path / "build_log" / "build.log").read_text(encoding="utf-8")
        assert ("job failed" if job_fails == "result" else "job exploded") in log
        assert "close exploded" in log
    elif mode == "queries":
        query_result = response["queries"][0]
        saved = json.loads((tmp_path / "queries" / "q1" / "result.json").read_text(encoding="utf-8"))
        assert saved == query_result
        result = query_result["answer_result"]
    if job_fails == "result":
        assert result["error"] == "job failed"
        assert result["metadata"] == {"diagnostic": "original"}
    else:
        assert result["error"] == "RuntimeError: job exploded"
        assert result["exception"]["type"] == "RuntimeError"
        assert "job exploded" in result["traceback"]
    assert result["job_call_counts"] == {"build": 1}
    assert result["token_usage"] == {}
    diagnostics = query_result["cleanup_errors"] if mode == "queries" else result["cleanup_errors"]
    assert diagnostics[0]["error"] == "ValueError: close exploded"
    assert diagnostics[0]["exception"]["type"] == "ValueError"
    assert "close exploded" in diagnostics[0]["traceback"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["config", "application"])
async def test_build_log_captures_initialization_failure(tmp_path, monkeypatch, failure_phase):
    if failure_phase == "config":

        def fail_config(_request):
            raise ValueError("invalid benchmark config")

        monkeypatch.setattr(sandbox_worker, "_prepare_runtime", fail_config)
        expected = "ValueError: invalid benchmark config"
    else:
        config = {
            "workspace_dir": str(tmp_path / "workspace"),
            "service": {"backend": "cli"},
            "jobs": {"broken": {"backend": "missing"}},
            "enable_logo": False,
            "log_to_console": False,
            "log_to_file": False,
        }
        monkeypatch.setattr(sandbox_worker, "_prepare_runtime", lambda _request: (tmp_path, config))
        expected = "ValueError: Unregistered backend 'missing' for Job 'broken'"

    with pytest.raises(ValueError, match="invalid benchmark config|Unregistered backend"):
        await sandbox_worker._run_build({"case_root": str(tmp_path), "jobs": [{"job": "build"}]})

    log_text = (tmp_path / "build_log" / "build.log").read_text(encoding="utf-8")
    assert "Traceback (most recent call last):" in log_text
    assert expected in log_text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", [None, "start", "checkpoint", "close", "start_and_close", "job"])
async def test_build_log_captures_runtime_failures_and_preserves_primary_error(tmp_path, monkeypatch, failure_phase):
    import reme.reme as reme_module
    from reme.utils import get_logger

    monkeypatch.setattr(sandbox_worker, "_prepare_runtime", lambda _request: (tmp_path, {}))
    startup_error = RuntimeError("startup exploded")
    checkpoint_error = RuntimeError("checkpoint exploded")
    close_error = RuntimeError("close exploded")
    closed = False

    class FakeReMe:
        def __init__(self, **_kwargs):
            get_logger(log_to_console=False, log_to_file=False, force_init=True)

        async def start(self):
            if failure_phase in {"start", "start_and_close"}:
                raise startup_error

        async def close(self):
            nonlocal closed
            closed = True
            if failure_phase in {"close", "start_and_close"}:
                raise close_error

    async def run_job(_app, _job, _arguments):
        if failure_phase == "job":
            return {"success": False, "error": "RuntimeError: job exploded", "traceback": "job traceback"}
        return {"success": True}

    def checkpoint(*_args):
        raise checkpoint_error

    monkeypatch.setattr(reme_module, "ReMe", FakeReMe)
    monkeypatch.setattr(sandbox_worker, "_run_job_on_app", run_job)
    monkeypatch.setattr(sandbox_worker, "_commit_memory_checkpoint", checkpoint)
    jobs = [{"job": "build"}]
    if failure_phase == "checkpoint":
        jobs[0]["memory_checkpoint"] = "boundary"
    request = {"case_root": str(tmp_path), "jobs": jobs}

    if failure_phase in {"start", "checkpoint", "close", "start_and_close"}:
        expected_error = checkpoint_error if failure_phase == "checkpoint" else startup_error
        if failure_phase == "close":
            expected_error = close_error
        with pytest.raises(RuntimeError) as failure:
            await sandbox_worker._run_build(request)
        assert failure.value is expected_error
    else:
        result = await sandbox_worker._run_build(request)
        assert result["success"] is (failure_phase is None)

    assert closed
    get_logger().info("after build")
    log_text = (tmp_path / "build_log" / "build.log").read_text(encoding="utf-8")
    assert "[sandbox] build phase started" in log_text
    assert "[sandbox] build phase finished" in log_text
    assert "after build" not in log_text
    if failure_phase in {"start", "checkpoint", "close", "start_and_close"}:
        assert "Traceback (most recent call last):" in log_text
        assert str(expected_error) in log_text
    if failure_phase == "start_and_close":
        assert "close exploded" in log_text
        assert any("close exploded" in note for note in failure.value.__notes__)
    if failure_phase == "job":
        assert "RuntimeError: job exploded" in log_text
        assert "job traceback" in log_text


@pytest.mark.asyncio
async def test_worker_clears_search_budget_between_queries(tmp_path, monkeypatch):
    import reme.reme as reme_module
    from reme.utils import get_logger

    monkeypatch.setattr(sandbox_worker, "_prepare_runtime", lambda _request: (tmp_path, {}))
    seen_metadata: list[dict] = []
    apps = []

    class FakeReMe:
        def __init__(self, **_kwargs):
            from types import SimpleNamespace

            self.context = SimpleNamespace(metadata={"shared": "retained"})
            apps.append(self)

        async def start(self):
            pass

        async def close(self):
            pass

    async def run_job(app, _job, _arguments):
        get_logger().info(f"answer body: {_arguments['query']}")
        seen_metadata.append(dict(app.context.metadata))
        app.context.metadata["tool_contexts"] = {"same-id": {"seen": True}}
        app.context.metadata["__search_call_budgets"] = {"same-id": {"count": 1}}
        return {"success": True, "answer": "ok", "metadata": {}, "error": None}

    monkeypatch.setattr(reme_module, "ReMe", FakeReMe)
    monkeypatch.setattr(sandbox_worker, "_run_job_on_app", run_job)

    result = await sandbox_worker._run_queries(
        {
            "case_root": str(tmp_path),
            "queries": [
                {"query_id": "q1", "question": "first"},
                {"query_id": "q2", "question": "second"},
            ],
        },
    )

    assert result["success"]
    assert len(apps) == 1
    assert seen_metadata == [{"shared": "retained"}, {"shared": "retained"}]
    assert apps[0].context.metadata == {"shared": "retained"}
    for query_id, own, other in (("q1", "first", "second"), ("q2", "second", "first")):
        log = (tmp_path / "queries" / query_id / "answer.log").read_text(encoding="utf-8")
        assert f"answer body: {own}" in log
        assert f"answer body: {other}" not in log


@pytest.mark.asyncio
async def test_worker_checkpoint_commits_only_memory_paths(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    (workspace / "daily").mkdir()
    (workspace / "daily" / "note.md").write_text("memory", encoding="utf-8")
    (workspace / "digest").mkdir()
    (workspace / "digest" / "note.md").write_text("digest", encoding="utf-8")
    (workspace / "sessions").mkdir()
    (workspace / "sessions" / "note.md").write_text("session", encoding="utf-8")
    (workspace / "other.txt").write_text("unrelated", encoding="utf-8")
    config = {"workspace_dir": str(workspace), "daily_dir": "daily", "digest_dir": "digest", "session_dir": "sessions"}
    monkeypatch.setattr(sandbox_worker, "_prepare_runtime", lambda _request: (tmp_path, config))

    result = await sandbox_worker._run_checkpoint({"message": "memory boundary"})

    assert result["success"] is True
    assert result["memory_checkpoint"]["message"] == "memory boundary"
    committed = subprocess.run(
        ["git", "show", "--pretty=format:", "--name-only", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    assert set(committed.stdout.splitlines()) == {"daily/note.md", "digest/note.md", "sessions/note.md"}
    assert result["memory_checkpoint"]["paths"] == ["daily", "digest", "sessions"]
    second = await sandbox_worker._run_checkpoint({"message": "empty boundary"})
    assert second["memory_checkpoint"]["commit_sha"] != result["memory_checkpoint"]["commit_sha"]

    config["daily_dir"] = "../outside"
    with pytest.raises(ValueError, match="escapes the runtime workspace"):
        await sandbox_worker._run_checkpoint({"message": "unsafe boundary"})


class FakeBackend:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commands: list[list[str]] = []
        self.worker_responses: dict[str, dict] = {}

    async def write_file(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def read_file(self, path: str) -> bytes:
        try:
            return self.files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    async def exec_shell(self, command: list[str], *, cwd=None, timeout=None):
        from types import SimpleNamespace

        self.commands.append(command)
        output = b""
        exit_code = 0
        if len(command) > 2 and command[1] == "-c":
            output = json.dumps(
                {
                    "reme": "/workspace/candidate/reme/__init__.py",
                    "agentscope": "2.0.9",
                    "plugins": {"lme": "/workspace/candidate/plugins/lme/src/reme_lme/__init__.py"},
                },
            ).encode()
        if "/workspace/harness/worker.py" in command:
            request = json.loads(self.files[command[command.index("--request") + 1]])
            mode = request.get("mode", "job")
            default = {"success": True, "jobs": []} if mode == "build" else {"success": True}
            response = self.worker_responses.get(mode, default)
            self.files[command[command.index("--response") + 1]] = json.dumps(response).encode()
            exit_code = 0 if response["success"] else 1
        if command[:2] == ["tar", "-czf"]:
            self.files[command[2]] = b"workspace archive"
        return SimpleNamespace(exit_code=exit_code, stdout=output, stderr=b"", ok=lambda: exit_code == 0)


class TimedOutBackend(FakeBackend):
    async def exec_shell(self, command: list[str], *, cwd=None, timeout=None):
        if "/workspace/harness/worker.py" in command:
            from types import SimpleNamespace

            self.commands.append(command)
            self.files["/workspace/case/build_log/build.log"] = b"build still running"
            return SimpleNamespace(exit_code=-1, stdout=b"", stderr=b"timed out", ok=lambda: False)
        return await super().exec_shell(command, cwd=cwd, timeout=timeout)


class FakeWorkspace:
    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend
        self.closed = False
        self.stopped = False

    async def initialize(self) -> None:
        pass

    def get_backend(self) -> FakeBackend:
        return self.backend

    async def close(self) -> None:
        self.closed = True

    async def stop(self) -> None:
        self.stopped = True


def _fake_docker_factory(tmp_path, workspace_builder, *, install_projects=(".",)):
    (tmp_path / "reme").mkdir()
    (tmp_path / "reme" / "__init__.py").write_text("ok", encoding="utf-8")
    return DockerReMeSandboxFactory(
        SourceCandidate(SourceSnapshot.from_directory(tmp_path), install_projects=install_projects),
        plugins=("lme",),
        workspace_builder=workspace_builder,
    )


@pytest.mark.asyncio
async def test_backend_timeout_prevents_direct_case_reset(tmp_path):
    workspace = FakeWorkspace(TimedOutBackend())
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case("timed-out")
    try:
        with pytest.raises(SandboxCommandTimeoutError):
            await case.run_build([JobRequest("build")])
        assert not case.reusable
        assert workspace.stopped and workspace.closed
        assert case.actions[-1].exit_code == -1
        assert await case.read_build_log() == "build still running"
        with pytest.raises(SandboxCommandError, match="no longer reusable"):
            await case.reset_case("next")
    finally:
        await case.close()
    assert workspace.closed


@pytest.mark.asyncio
async def test_external_timeout_prevents_case_reset(tmp_path):
    started = asyncio.Event()

    class BlockingBackend(FakeBackend):
        async def exec_shell(self, command, *, cwd=None, timeout=None):
            if "/workspace/harness/worker.py" in command:
                self.commands.append(command)
                started.set()
                await asyncio.Event().wait()
            return await super().exec_shell(command, cwd=cwd, timeout=timeout)

    workspace = FakeWorkspace(BlockingBackend())
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case("timed-out")
    try:
        build = asyncio.create_task(case.run_build([JobRequest("build")]))
        await asyncio.wait_for(started.wait(), timeout=2)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(build, timeout=0.01)
        assert not case.reusable
        assert workspace.stopped and workspace.closed
        with pytest.raises(SandboxCommandError, match="no longer reusable"):
            await case.reset_case("next")
        inbox_count = sum("/inbox/" in path for path in workspace.backend.files)
        with pytest.raises(SandboxCommandError, match="no longer reusable"):
            await case.run_build([JobRequest("build")])
        assert sum("/inbox/" in path for path in workspace.backend.files) == inbox_count
    finally:
        await case.close()
    assert workspace.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["deadline", "backend-timeout", "cancel"])
async def test_interrupted_worker_stops_before_reading_logs_and_removal(tmp_path, failure):
    events = []
    started = asyncio.Event()
    stopping = asyncio.Event()
    release_stop = asyncio.Event()

    class BlockingBackend(FakeBackend):
        async def exec_shell(self, command, *, cwd=None, timeout=None):
            if "/workspace/harness/worker.py" in command:
                self.files["/workspace/case/build_log/build.log"] = b"closing application"
                started.set()
                if failure == "backend-timeout":
                    raise TimeoutError("backend timed out")
                await asyncio.Event().wait()  # Includes a worker stuck in Application.close().
            return await super().exec_shell(command, cwd=cwd, timeout=timeout)

        async def read_file(self, path):
            events.append("read")
            assert workspace.stopped
            return await super().read_file(path)

    class OrderedWorkspace(FakeWorkspace):
        async def stop(self):
            stopping.set()
            await release_stop.wait()
            events.append("stop")
            await super().stop()

        async def close(self):
            events.append("close")
            await super().close()

    workspace = OrderedWorkspace(BlockingBackend())
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case("interrupted")
    case.command_timeout = 0.01 if failure == "deadline" else 10
    task = asyncio.create_task(case.run_build([JobRequest("build")]))
    await asyncio.wait_for(started.wait(), timeout=2)
    if failure == "cancel":
        task.cancel()
    await asyncio.wait_for(stopping.wait(), timeout=2)
    if failure == "cancel":
        task.cancel()  # Repeated cancellation must not interrupt container cleanup.
    release_stop.set()
    expected = asyncio.CancelledError if failure == "cancel" else SandboxCommandTimeoutError
    with pytest.raises(expected):
        await asyncio.wait_for(task, timeout=2)
    assert events == ["stop", "read", "close"]
    assert workspace.closed and not case.reusable
    assert await case.read_build_log() == "closing application"
    await case.close()
    assert events == ["stop", "read", "close"]


@pytest.mark.asyncio
async def test_failed_stop_still_removes_container_and_preserves_timeout(tmp_path):
    class BrokenStopWorkspace(FakeWorkspace):
        async def stop(self):
            raise RuntimeError("kill failed")

    workspace = BrokenStopWorkspace(TimedOutBackend())
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case("failed-stop")
    with pytest.raises(SandboxCommandTimeoutError) as failure:
        await case.run_build([JobRequest("build")])
    assert workspace.closed
    assert any("kill failed" in note for note in failure.value.__notes__)
    with pytest.raises(SandboxCommandError, match="log unavailable"):
        await case.read_build_log()


@pytest.mark.asyncio
async def test_blocked_diagnostics_cannot_prevent_removal(tmp_path, monkeypatch):
    from deploy.benchmark_docker import docker_sandbox

    monkeypatch.setattr(docker_sandbox, "_LOG_TIMEOUT", 0.01)

    class BlockedLogBackend(TimedOutBackend):
        async def read_file(self, path):
            assert workspace.stopped
            await asyncio.Event().wait()

    workspace = FakeWorkspace(BlockedLogBackend())
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case("blocked-log")
    with pytest.raises(SandboxCommandTimeoutError):
        await asyncio.wait_for(case.run_build([JobRequest("build")]), timeout=2)
    assert workspace.closed


@pytest.mark.asyncio
async def test_cancelled_request_upload_prevents_case_reset(tmp_path):
    started = asyncio.Event()

    class BlockingBackend(FakeBackend):
        async def write_file(self, path, data):
            if "/inbox/" in path:
                started.set()
                await asyncio.Event().wait()
            await super().write_file(path, data)

    workspace = FakeWorkspace(BlockingBackend())
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case("cancelled-upload")
    try:
        build = asyncio.create_task(case.run_build([JobRequest("build")]))
        await asyncio.wait_for(started.wait(), timeout=2)
        build.cancel()
        with pytest.raises(asyncio.CancelledError):
            await build
        assert not case.reusable
        with pytest.raises(SandboxCommandError, match="no longer reusable"):
            await case.reset_case("next")
    finally:
        await case.close()
    assert workspace.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["write", "exec"])
async def test_backend_error_prevents_case_reset(tmp_path, operation):
    class FailingBackend(FakeBackend):
        async def write_file(self, path, data):
            if operation == "write" and "/inbox/" in path:
                raise RuntimeError("Docker stream failed")
            await super().write_file(path, data)

        async def exec_shell(self, command, *, cwd=None, timeout=None):
            if operation == "exec" and "/workspace/harness/worker.py" in command:
                raise RuntimeError("Docker stream failed")
            return await super().exec_shell(command, cwd=cwd, timeout=timeout)

    workspace = FakeWorkspace(FailingBackend())
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case(f"{operation}-error")
    try:
        with pytest.raises(RuntimeError, match="Docker stream failed"):
            await case.run_build([JobRequest("build")])
        assert not case.reusable
        with pytest.raises(SandboxCommandError, match="no longer reusable"):
            await case.reset_case("next")
    finally:
        await case.close()
    assert workspace.closed


@pytest.mark.asyncio
async def test_cancelled_close_can_be_retried(tmp_path):
    started = asyncio.Event()

    class InterruptibleWorkspace(FakeWorkspace):
        def __init__(self):
            super().__init__(FakeBackend())
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                started.set()
                await asyncio.Event().wait()
            self.closed = True

    workspace = InterruptibleWorkspace()
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case("close-retry")
    closing = asyncio.create_task(case.close())
    await asyncio.wait_for(started.wait(), timeout=2)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert not case.reusable
    assert not workspace.closed

    await case.close()
    assert workspace.closed
    assert workspace.close_calls == 2


@pytest.mark.asyncio
async def test_source_candidate_installs_plugin_and_closes_container(tmp_path):
    backend = FakeBackend()
    workspace = FakeWorkspace(backend)
    factory = _fake_docker_factory(
        tmp_path,
        lambda _id, _image, _env: workspace,
        install_projects=(".", "plugins/lme"),
    )
    case = await factory.create_case("example")
    try:
        assert any(command[-1] == "/workspace/candidate/plugins/lme" for command in backend.commands)
        assert "/workspace/candidate.tar.gz" in backend.files
        assert "/workspace/harness/worker.py" in backend.files
        assert (await case.run_build([JobRequest("auto_memory")]))["success"]
        exported = await case.export_workspace(tmp_path / "workspace.tar.gz")
        assert exported.read_bytes() == b"workspace archive"
        with pytest.raises(ValueError, match="private"):
            await case.run_queries(
                [EvaluationQuery("q1", "question", "secret", answer_arguments={"golden_answer": "secret"})],
            )
    finally:
        await case.close()
    assert workspace.closed


@pytest.mark.asyncio
async def test_job_and_checkpoint_share_worker_request_path(tmp_path):
    backend = FakeBackend()
    workspace = FakeWorkspace(backend)
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case("example")
    try:
        result = await case.run_job("answer", {"query": "hello"})
        assert result.success
        assert case.actions[-1].name == "job:answer"
        job_request = json.loads(backend.files[next(path for path in backend.files if "/inbox/" in path)])
        assert job_request["mode"] == "job"
        assert job_request["arguments"] == {"query": "hello"}

        await case.commit_memory_history("memory boundary")
        assert case.actions[-1].name == "worker:checkpoint"
        requests = [json.loads(content) for path, content in backend.files.items() if "/inbox/" in path]
        assert requests[-1]["mode"] == "checkpoint"
        assert requests[-1]["message"] == "memory boundary"
        assert not any(command[0] == "git" and "commit" in command for command in backend.commands)
        with pytest.raises(ValueError, match="non-empty"):
            await case.commit_memory_history(" ")

        backend.worker_responses["checkpoint"] = {"success": False, "error": "git commit failed"}
        with pytest.raises(SandboxCommandError, match="git commit failed"):
            await case.commit_memory_history("next boundary")
        assert case.actions[-1].exit_code == 1
    finally:
        await case.close()


@pytest.mark.asyncio
async def test_pool_replaces_worker_after_backend_timeout(tmp_path):
    timed_out_workspace = FakeWorkspace(TimedOutBackend())
    replacement_workspace = FakeWorkspace(FakeBackend())
    workspaces = iter((timed_out_workspace, replacement_workspace))
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: next(workspaces))
    plans = [
        CasePlan("timed-out", build_jobs=(JobRequest("build"),), export_build_log_to=tmp_path / "build.log"),
        CasePlan("next", build_jobs=(JobRequest("build"),)),
    ]

    results = await BenchmarkDockerPool(factory, concurrency=1).run(plans)

    assert "SandboxCommandTimeoutError" in results[0].error
    assert results[0].build_log == "build still running"
    assert (tmp_path / "build.log").read_text(encoding="utf-8") == "build still running"
    assert results[1].success
    assert timed_out_workspace.closed and replacement_workspace.closed
    assert sum(command[:2] == ["rm", "-rf"] for command in timed_out_workspace.backend.commands) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "required", "description", "members"),
    [
        (
            "export_memory_construction",
            "build_log/build.log",
            "memory construction export",
            ["manifest.json", "reme_workspace", "build_log"],
        ),
        ("export_queries", "queries/summary.json", "query export", ["manifest.json", "queries"]),
        ("export_query", "queries/q1/result.json", "query export", ["manifest.json", "queries/q1"]),
        (
            "export_evaluation",
            "queries/summary.json",
            "evaluation export",
            ["manifest.json", "reme_workspace", "queries"],
        ),
    ],
)
async def test_archive_exports_check_required_artifacts(tmp_path, method, required, description, members):
    backend = FakeBackend()
    case = await _fake_docker_factory(tmp_path, lambda _id, _image, _env: FakeWorkspace(backend)).create_case("example")
    destination = tmp_path / "exports" / "result.tar.gz"
    export = getattr(case, method)
    args = ("q1", destination) if method == "export_query" else (destination,)
    try:
        with pytest.raises(SandboxCommandError) as failure:
            await export(*args)
        assert str(failure.value) == f"{description} is missing required artifact: /workspace/case/{required}"
        backend.files[f"/workspace/case/{required}"] = b"{}"
        assert await export(*args) == destination
        assert ["tar", "-czf", "/tmp/reme-case-export.tar.gz", "-C", "/workspace/case", *members] in backend.commands
    finally:
        await case.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    ["export_workspace", "export_memory_construction", "export_evaluation"],
)
async def test_exported_memory_can_seed_queries_without_build(tmp_path, method):
    class ArchiveBackend(FakeBackend):
        async def exec_shell(self, command, *, cwd=None, timeout=None):
            result = await super().exec_shell(command, cwd=cwd, timeout=timeout)
            if command[:2] == ["tar", "-czf"]:
                root = command[command.index("-C") + 1] + "/"
                members = command[command.index("-C") + 2 :]
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                    for path, content in self.files.items():
                        if not path.startswith(root):
                            continue
                        name = path[len(root) :]
                        if not any(
                            member == "." or name == member or name.startswith(member + "/") for member in members
                        ):
                            continue
                        info = tarfile.TarInfo(name)
                        info.size = len(content)
                        archive.addfile(info, io.BytesIO(content))
                self.files[command[2]] = stream.getvalue()
            return result

    source_backend, query_backend = ArchiveBackend(), FakeBackend()
    workspaces = iter((FakeWorkspace(source_backend), FakeWorkspace(query_backend)))
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: next(workspaces))
    source = await factory.create_case("build")
    query_case = await factory.create_case("query")
    try:
        assert (await source.run_build([JobRequest("build")]))["success"]
        source_backend.files.update(
            {
                "/workspace/case/reme_workspace/daily/note.md": b"saved memory",
                "/workspace/case/reme_workspace/.git/HEAD": b"ref: refs/heads/main\n",
                "/workspace/case/build_log/build.log": b"build log",
                "/workspace/case/queries/summary.json": b"{}",
            },
        )
        saved = await getattr(source, method)(tmp_path / "memory.tar.gz")
        await query_case.upload_workspace(saved)
        uploaded = query_backend.files["/tmp/reme-workspace-upload.tar.gz"]
        with tarfile.open(fileobj=io.BytesIO(uploaded), mode="r:gz") as archive:
            assert set(archive.getnames()) == {"daily/note.md", ".git/HEAD"}
            assert archive.extractfile("daily/note.md").read() == b"saved memory"
        assert (await query_case.run_queries([EvaluationQuery("q1", "Question?", "Answer")]))["success"]
        requests = [json.loads(content) for path, content in query_backend.files.items() if "/inbox/" in path]
        assert [request["mode"] for request in requests] == ["queries"]
    finally:
        await source.close()
        await query_case.close()


@pytest.mark.parametrize(
    ("manifest", "workspace_path", "error"),
    [
        (None, "reme_workspace/daily/note.md", "unmarked case archive"),
        ({"archive_format": "reme-case-v1", "workspace_root": None}, None, "no reusable workspace"),
        ({"archive_format": "reme-case-v1", "workspace_root": "../escape"}, None, "workspace root"),
        ({"archive_format": "reme-case-v2", "workspace_root": "reme_workspace"}, None, "unsupported.*format"),
        ({"archive_format": "reme-case-v1", "workspace_root": "reme_workspace"}, None, "workspace is missing"),
        ({}, "reme_workspace/daily/note.md", None),
    ],
)
def test_workspace_import_checks_case_archive_format(tmp_path, manifest, workspace_path, error):
    source = tmp_path / "case.tar.gz"
    files = {"queries/q1/result.json": b"{}"}
    if manifest is not None:
        files["manifest.json"] = json.dumps(manifest).encode()
    if workspace_path is not None:
        files[workspace_path] = b"memory"
    with tarfile.open(source, "w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    if error is not None:
        with pytest.raises(ValueError, match=error):
            _archive_workspace(source)
    else:
        with tarfile.open(fileobj=io.BytesIO(_archive_workspace(source)), mode="r:gz") as archive:
            assert archive.getnames() == ["daily/note.md"]


@pytest.mark.asyncio
async def test_default_analysis_export_saves_logs_outside_archive(tmp_path):
    class AnalysisBackend(FakeBackend):
        async def exec_shell(self, command, *, cwd=None, timeout=None):
            result = await super().exec_shell(command, cwd=cwd, timeout=timeout)
            if command[:2] == ["tar", "-czf"] and command[-1] == "queries":
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                    for name, content in (
                        ("queries/summary.json", b"{}"),
                        ("queries/q1/answer.log", b"answer log"),
                    ):
                        info = tarfile.TarInfo(name)
                        info.size = len(content)
                        archive.addfile(info, io.BytesIO(content))
                self.files[command[2]] = stream.getvalue()
            return result

    backend = AnalysisBackend()
    workspace = FakeWorkspace(backend)
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    case = await factory.create_case("example")
    try:
        backend.files["/workspace/case/runtime-layout.json"] = (
            b'{"workspace_root":"reme_workspace","analysis_excludes":[]}'
        )
        backend.files["/workspace/case/build_log/build.log"] = b"build log"
        backend.files["/workspace/case/queries/summary.json"] = b"{}"
        archive = tmp_path / "analysis.tar.gz"
        assert await case.export(archive) == archive
        assert archive.read_bytes() == b"workspace archive"
        artifacts = tmp_path / "analysis.tar.gz.artifacts"
        assert (artifacts / "build_log" / "build.log").read_bytes() == b"build log"
        assert (artifacts / "queries" / "summary.json").read_bytes() == b"{}"
        assert (artifacts / "queries" / "q1" / "answer.log").read_bytes() == b"answer log"
        archive_command = next(
            command for command in backend.commands if command[:3] == ["tar", "-czf", "/tmp/reme-case-export.tar.gz"]
        )
        assert "build_log" not in archive_command
        assert "queries" not in archive_command
    finally:
        await case.close()


@pytest.mark.asyncio
async def test_cancelled_case_initialization_closes_workspace(tmp_path):
    started = asyncio.Event()

    class BlockingWorkspace(FakeWorkspace):
        async def initialize(self):
            started.set()
            await asyncio.Event().wait()

    workspace = BlockingWorkspace(FakeBackend())
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    task = asyncio.create_task(factory.create_case("cancelled"))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert workspace.closed


@pytest.mark.asyncio
async def test_cancelled_batch_creation_closes_completed_workspace_and_reports_cleanup_failure(tmp_path):
    second_started = asyncio.Event()

    class BlockingWorkspace(FakeWorkspace):
        async def initialize(self):
            second_started.set()
            await asyncio.Event().wait()

        async def close(self):
            await super().close()
            raise RuntimeError("delete failed")

    first = FakeWorkspace(FakeBackend())
    second = BlockingWorkspace(FakeBackend())
    workspaces = iter((first, second))
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: next(workspaces))
    task = asyncio.create_task(factory.create_cases(["first", "second"], concurrency=2))
    await asyncio.wait_for(second_started.wait(), timeout=2)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as failure:
        await task
    assert first.closed and second.closed
    assert any("delete failed" in note for note in failure.value.__notes__)


@pytest.mark.asyncio
async def test_failed_case_creation_preserves_initialization_error_when_close_fails(tmp_path):
    class BrokenWorkspace(FakeWorkspace):
        async def initialize(self):
            raise RuntimeError("initialization failed")

        async def close(self):
            raise RuntimeError("close failed")

    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: BrokenWorkspace(FakeBackend()))
    with pytest.raises(RuntimeError, match="initialization failed") as failure:
        await factory.create_case("broken")
    assert any("close failed" in note for note in failure.value.__notes__)


@pytest.mark.asyncio
async def test_failed_batch_creation_preserves_creation_error_when_close_fails(tmp_path):
    class CloseFailsWorkspace(FakeWorkspace):
        async def close(self):
            raise RuntimeError("close failed")

    class InitializeFailsWorkspace(FakeWorkspace):
        async def initialize(self):
            raise RuntimeError("initialization failed")

    workspaces = iter((CloseFailsWorkspace(FakeBackend()), InitializeFailsWorkspace(FakeBackend())))
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: next(workspaces))
    with pytest.raises(RuntimeError, match="initialization failed") as failure:
        await factory.create_cases(["first", "second"], concurrency=2)
    assert any("close failed" in note for note in failure.value.__notes__)


@pytest.mark.asyncio
async def test_factory_finishes_failed_creation_cleanup_before_cancellation(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()

    class BrokenWorkspace(FakeWorkspace):
        async def initialize(self):
            raise RuntimeError("initialization failed")

        async def close(self):
            started.set()
            await release.wait()
            self.closed = True

    workspace = BrokenWorkspace(FakeBackend())
    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: workspace)
    task = asyncio.create_task(factory.create_case("failed"))
    await asyncio.wait_for(started.wait(), timeout=2)
    for _ in range(2):
        task.cancel()
        await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError) as failure:
        await asyncio.wait_for(task, timeout=2)
    assert workspace.closed
    assert "initialization failed" in str(failure.value.__cause__)


@pytest.mark.asyncio
async def test_batch_reports_all_initialization_and_cleanup_failures(tmp_path):
    class BrokenWorkspace(FakeWorkspace):
        async def initialize(self):
            raise RuntimeError("initialization failed")

        async def close(self):
            raise RuntimeError("delete failed")

    factory = _fake_docker_factory(tmp_path, lambda _id, _image, _env: BrokenWorkspace(FakeBackend()))
    with pytest.raises(RuntimeError, match="initialization failed") as failure:
        await factory.create_cases(["first", "second"], concurrency=2)
    assert any("additional creation failure" in note for note in failure.value.__notes__)
    assert sum("delete failed" in note for note in failure.value.__notes__) == 2


@pytest.mark.asyncio
async def test_pool_retries_strict_delete_without_replacing_uncertain_container(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    pytest.importorskip("aiodocker")
    from deploy.benchmark_docker.direct_docker_workspace import DirectDockerWorkspace

    workspaces = []
    container = SimpleNamespace(
        kill=AsyncMock(),
        wait=AsyncMock(),
        delete=AsyncMock(side_effect=[RuntimeError("delete failed"), RuntimeError("delete failed"), None]),
    )
    client = SimpleNamespace(close=AsyncMock())

    class StrictWorkspace(DirectDockerWorkspace):
        # Provisioning deliberately populates the inherited Docker handles.
        # pylint: disable=attribute-defined-outside-init
        async def _provision_backend(self):
            self._container, self._client, self._backend = container, client, TimedOutBackend()

    def build_workspace(workspace_id, image, env):
        workspace = StrictWorkspace(workspace_id=workspace_id, base_image=image, env=env)
        workspaces.append(workspace)
        return workspace

    factory = _fake_docker_factory(tmp_path, build_workspace)
    plans = [CasePlan(case_id, build_jobs=(JobRequest("build"),)) for case_id in ("timed-out", "waiting")]
    results = await asyncio.wait_for(BenchmarkDockerPool(factory, concurrency=1).run(plans), timeout=2)
    assert not any(result.success for result in results)
    assert "worker close failed" in results[0].error
    assert "delete failed" in results[1].error
    assert len(workspaces) == 1
    assert container.delete.await_count == 3
    assert workspaces[0]._container is workspaces[0]._client is None
    client.close.assert_awaited_once()
