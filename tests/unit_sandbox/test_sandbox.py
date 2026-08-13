"""Tests for the standalone Docker benchmark sandbox infrastructure."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from reme.enumeration import ComponentEnum
from reme.schema import ApplicationConfig
from reme.utils import global_counter_add
from sandbox import (
    DockerReMeSandboxFactory,
    EvaluationQuery,
    ImageCandidate,
    JobRequest,
    SourceCandidate,
    SourceSnapshot,
)
from sandbox.direct_docker_workspace import DirectDockerWorkspace
from sandbox import worker


class FakeExecResult:
    """Minimal AgentScope ExecResult stand-in."""

    def __init__(self, exit_code=0, stdout=b"", stderr=b""):
        """Store a fake process result."""
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

    def ok(self):
        """Return whether the fake command succeeded."""
        return self.exit_code == 0


class FakeBackend:
    """In-memory backend that emulates the worker and export boundaries."""

    def __init__(self):
        """Initialize an empty in-memory filesystem and command log."""
        self.files = {}
        self.commands = []

    async def write_file(self, path, data):
        """Write bytes into the fake filesystem."""
        self.files[path] = data

    async def read_file(self, path):
        """Read bytes from the fake filesystem."""
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def exec_shell(self, command, *, cwd=None, timeout=None):
        """Record a command and emulate validation, worker, and tar output."""
        self.commands.append((command, cwd, timeout))
        if command[:2] == ["rm", "-rf"]:
            roots = tuple(f"{path}/" for path in command[2:])
            for path in list(self.files):
                if path in command[2:] or path.startswith(roots):
                    del self.files[path]
        if command[:2] == ["git", "init"]:
            self.files[f"{cwd}/.git/HEAD"] = b"ref: refs/heads/master\n"
        if command[:2] == ["python3", "-c"] or (len(command) > 2 and command[1:3] == ["-c", command[2]]):
            stdout = b'{"reme": "/workspace/candidate/reme/__init__.py", "agentscope": "2.0.4.post1"}\n'
            return FakeExecResult(stdout=stdout)
        if "/workspace/harness/worker.py" in command:
            request_path = command[command.index("--request") + 1]
            response_path = command[command.index("--response") + 1]
            request = json.loads(self.files[request_path])
            self.files["/workspace/case/runtime-layout.json"] = json.dumps(
                {
                    "workspace_root": "reme_workspace",
                    "configured_paths": {
                        "metadata_dir": "reme_workspace/metadata",
                        "session_dir": "reme_workspace/session",
                        "resource_dir": "reme_workspace/resource",
                        "dialog_dir": "reme_workspace/session/dialog",
                        "daily_dir": "reme_workspace/daily",
                        "digest_dir": "reme_workspace/digest",
                        "mem_session_dir": "reme_workspace/mem_session",
                    },
                    "analysis_excludes": ["reme_workspace/metadata", "reme_workspace/resource"],
                },
            ).encode()
            if request.get("mode") == "build":
                self.files["/workspace/case/build_log/build.log"] = b"build"
                response = {
                    "success": True,
                    "jobs": [{"job": item["job"], "result": {"success": True}} for item in request["jobs"]],
                }
            elif request.get("mode") == "queries":
                self.files["/workspace/case/queries/summary.json"] = b'{"queries": []}'
                response = {
                    "success": True,
                    "summary": {
                        "queries": [{"query_id": item["query_id"], "score": 1.0} for item in request["queries"]],
                    },
                    "queries": [],
                }
            elif request.get("mode") == "query":
                item = request["queries"][0]
                query_root = f"/workspace/case/queries/{item['query_id']}"
                result = {
                    "query_id": item["query_id"],
                    "question": item["question"],
                    "golden_answer": item.get("golden_answer"),
                    "answer": item.get("golden_answer"),
                    "score": 1.0,
                    "error": None,
                }
                self.files[f"{query_root}/result.json"] = json.dumps(result).encode()
                response = {
                    "success": True,
                    "summary": {"queries": [{"query_id": item["query_id"], "score": 1.0}]},
                    "queries": [result],
                }
            else:
                response = {
                    "success": True,
                    "answer": "yes",
                    "metadata": {"job": "ok"},
                    "token_usage": {
                        "bench": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
                    },
                    "job_call_counts": {"answer_judge": 1},
                    "error": None,
                }
            self.files[response_path] = json.dumps(response).encode()
        if command[:2] == ["tar", "-czf"]:
            self.files[command[2]] = b"case-archive"
        return FakeExecResult()


class FakeWorkspace:
    """A distinct workspace object for each factory call."""

    def __init__(self, case_id, image, env, backend=None):
        """Initialize one unique fake workspace."""
        self.case_id = case_id
        self.image = image
        self.env = env
        self.backend = backend or FakeBackend()
        self.initialized = False
        self.closed = False

    async def initialize(self):
        """Mark the fake workspace initialized."""
        self.initialized = True

    def get_backend(self):
        """Return the fake backend."""
        return self.backend

    async def close(self):
        """Mark the fake workspace closed."""
        self.closed = True


@pytest.mark.asyncio
async def test_direct_workspace_uses_supplied_image_without_gateway_build(monkeypatch):
    """The direct backend starts Docker without deriving an MCP gateway image."""
    workspace = DirectDockerWorkspace(base_image="candidate:exact")
    backend = FakeBackend()

    async def provision():
        """Bind a fake backend as Docker provisioning would."""
        workspace._backend = backend  # pylint: disable=protected-access

    monkeypatch.setattr(workspace, "_provision_backend", provision)
    await workspace._build_or_reuse_image()  # pylint: disable=protected-access
    await workspace.initialize()

    assert workspace._image_tag == "candidate:exact"  # pylint: disable=protected-access
    assert workspace.is_alive is True
    assert backend.commands == [(["mkdir", "-p", "/workspace"], "/", None)]


def test_source_snapshot_is_deterministic_and_excludes_runtime_state(tmp_path):
    """Equivalent source trees produce identical safe archives."""
    (tmp_path / "reme").mkdir()
    (tmp_path / "reme" / "main.py").write_text("print('candidate')\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret-ish vcs state")
    (tmp_path / ".env").write_text("TOKEN=secret")

    first = SourceSnapshot.from_directory(tmp_path)
    second = SourceSnapshot.from_directory(tmp_path)

    assert first.archive == second.archive
    assert first.sha256 == second.sha256
    assert first.file_count == 1
    with tarfile.open(fileobj=io.BytesIO(first.archive), mode="r:gz") as archive:
        assert archive.getnames() == ["reme/main.py"]


def test_source_snapshot_rejects_symlinks(tmp_path):
    """A source snapshot never follows links outside the candidate root."""
    target = tmp_path / "target.py"
    target.write_text("pass\n")
    (tmp_path / "link.py").symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        SourceSnapshot.from_directory(tmp_path)


@pytest.mark.asyncio
async def test_one_source_candidate_is_reused_in_independent_case_workspaces(tmp_path):
    """The factory reuses code bytes but creates distinct containers."""
    (tmp_path / "reme").mkdir()
    (tmp_path / "reme" / "__init__.py").write_text("__version__ = 'candidate'\n")
    (tmp_path / "pyproject.toml").write_text("[build-system]\nrequires=[]\nbuild-backend='x'\n")
    candidate = SourceCandidate(SourceSnapshot.from_directory(tmp_path), base_image="base:test")
    workspaces = []

    def build(case_id, image, env):
        """Capture every separately constructed workspace."""
        workspace = FakeWorkspace(case_id, image, env)
        workspaces.append(workspace)
        return workspace

    factory = DockerReMeSandboxFactory(candidate, env={"LLM_API_KEY": "secret"}, workspace_builder=build)
    first, second = await factory.create_cases(["case-1", "case-2"], concurrency=2)

    assert first.workspace is not second.workspace
    assert first.container_id != second.container_id
    assert first.container_id.startswith("reme-benchmark-")
    assert second.container_id.startswith("reme-benchmark-")
    assert workspaces[0].case_id == first.container_id
    assert workspaces[1].case_id == second.container_id
    assert "case-1" not in first.container_id
    assert "case-2" not in second.container_id
    assert workspaces[0].image == workspaces[1].image == "base:test"
    assert workspaces[0].backend.files["/workspace/candidate.tar.gz"] is candidate.snapshot.archive
    assert workspaces[1].backend.files["/workspace/candidate.tar.gz"] is candidate.snapshot.archive
    assert first.runtime_workspace == second.runtime_workspace == "/workspace/case/reme_workspace"
    assert all(
        not any("reme start" in part for part in command) for ws in workspaces for command, _, _ in ws.backend.commands
    )


@pytest.mark.asyncio
async def test_image_candidate_skips_source_upload_and_runs_jobs(tmp_path):
    """An immutable image candidate needs no per-case source transfer."""
    workspaces = []

    def build(case_id, image, env):
        """Capture the workspace built for the image candidate."""
        workspace = FakeWorkspace(case_id, image, env)
        workspaces.append(workspace)
        return workspace

    factory = DockerReMeSandboxFactory(ImageCandidate("reme-candidate:test"), workspace_builder=build)
    case = await factory.create_case("case-image")
    result = await case.run_job("answer_judge", {"query": "q"})
    exported = await case.export(tmp_path / "case.tar.gz")

    assert result.success is True
    assert result.answer == "yes"
    assert result.token_usage == {
        "bench": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
    }
    assert result.job_call_counts == {"answer_judge": 1}
    assert "/workspace/candidate.tar.gz" not in workspaces[0].backend.files
    assert exported.read_bytes() == b"case-archive"
    manifest = json.loads(workspaces[0].backend.files["/workspace/case/manifest.json"])
    assert manifest["candidate_mode"] == "image"
    assert manifest["container_id"] == case.container_id
    export_command = next(command for command, _, _ in workspaces[0].backend.commands if command[:2] == ["tar", "-czf"])
    assert "logs" in export_command
    assert "results" in export_command
    assert "reme_workspace" in export_command
    assert "inbox" not in export_command
    assert "--exclude=reme_workspace/metadata" in export_command
    assert "--exclude=reme_workspace/resource" in export_command
    assert "--exclude=reme_workspace/session" not in export_command


@pytest.mark.asyncio
async def test_build_and_queries_use_batch_worker_requests(tmp_path):
    """Build and query batches each use one worker process and preserve verbatim query IDs."""
    workspace = FakeWorkspace("case-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")

    build = await case.run_build(
        [
            JobRequest("auto_memory", {"session_id": "session-1"}),
            JobRequest("index_update"),
        ],
    )
    queries = await case.run_queries(
        [
            EvaluationQuery(
                query_id="information_extraction:1",
                question="What happened?",
                golden_answer="An event",
                judge_arguments={"query": "What happened?", "golden_answer": "An event"},
            ),
            EvaluationQuery(
                query_id="information_extraction:2",
                question="When?",
                golden_answer="Yesterday",
                judge_arguments={"query": "When?", "golden_answer": "Yesterday"},
            ),
        ],
    )

    assert build["success"] is True
    assert [item["query_id"] for item in queries["summary"]["queries"]] == [
        "information_extraction:1",
        "information_extraction:2",
    ]
    worker_commands = [
        command for command, _, _ in workspace.backend.commands if "/workspace/harness/worker.py" in command
    ]
    assert len(worker_commands) == 2

    construction = await case.export_memory_construction(tmp_path / "memory-construction.tar.gz")
    assert construction.read_bytes() == b"case-archive"
    assert workspace.backend.commands[-1][0] == [
        "tar",
        "-czf",
        "/tmp/reme-case-export.tar.gz",
        "-C",
        "/workspace/case",
        "reme_workspace",
        "build_log",
    ]

    build_log = await case.export_build_log(tmp_path / "build.log")
    assert build_log.read_bytes() == b"build"

    query_artifacts = await case.export_queries(tmp_path / "queries.tar.gz")
    assert query_artifacts.read_bytes() == b"case-archive"
    assert workspace.backend.commands[-1][0] == [
        "tar",
        "-czf",
        "/tmp/reme-case-export.tar.gz",
        "-C",
        "/workspace/case",
        "queries",
    ]

    exported = await case.export_evaluation(tmp_path / "evaluation.tar.gz")
    assert exported.read_bytes() == b"case-archive"
    export_command = workspace.backend.commands[-1][0]
    assert export_command == [
        "tar",
        "-czf",
        "/tmp/reme-case-export.tar.gz",
        "-C",
        "/workspace/case",
        "reme_workspace",
        "build_log",
        "queries",
    ]


@pytest.mark.asyncio
async def test_single_query_api_appends_and_exports_one_lease(tmp_path):
    """The lease primitive does not require or create a case-level summary."""

    workspace = FakeWorkspace("case-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")
    query = EvaluationQuery(query_id="query:1", question="What?", golden_answer="This")

    result = await case.run_query(query)
    exported = await case.export_query(query.query_id, tmp_path / "query.tar.gz")

    assert result["query_id"] == "query:1"
    assert result["score"] == 1.0
    assert "/workspace/case/queries/summary.json" not in workspace.backend.files
    assert exported.read_bytes() == b"case-archive"
    assert workspace.backend.commands[-1][0] == [
        "tar",
        "-czf",
        "/tmp/reme-case-export.tar.gz",
        "-C",
        "/workspace/case",
        "queries/query:1",
    ]


@pytest.mark.asyncio
async def test_query_artifact_ids_reject_path_components():
    """Verbatim query directory names cannot escape the queries root."""
    workspace = FakeWorkspace("case-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")

    with pytest.raises(ValueError, match="safe directory"):
        await case.run_queries(
            [EvaluationQuery(query_id="../escape", question="q", golden_answer="a")],
        )

    with pytest.raises(ValueError, match="conflicts"):
        await case.run_queries(
            [EvaluationQuery(query_id="summary.json", question="q", golden_answer="a")],
        )


@pytest.mark.asyncio
async def test_workspace_can_be_cleared_and_uploaded_from_host(tmp_path):
    """A completed memory can be copied into an independent query case."""
    source = tmp_path / "built-memory"
    (source / "daily").mkdir(parents=True)
    (source / "daily" / "2026-08-06.md").write_text("remember this\n")
    workspace = FakeWorkspace("query-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("query-1")
    old_path = f"{case.runtime_workspace}/daily/old.md"
    workspace.backend.files[old_path] = b"old memory"

    await case.upload_workspace(source)

    payload = workspace.backend.files["/tmp/reme-workspace-upload.tar.gz"]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert archive.getnames() == ["daily", "daily/2026-08-06.md"]
        assert archive.extractfile("daily/2026-08-06.md").read() == b"remember this\n"
    clear_index = next(
        index
        for index, (command, _, _) in enumerate(workspace.backend.commands)
        if command == ["rm", "-rf", case.runtime_workspace]
    )
    upload_index = next(
        index
        for index, (command, _, _) in enumerate(workspace.backend.commands)
        if command == ["tar", "-xzf", "/tmp/reme-workspace-upload.tar.gz", "-C", case.runtime_workspace]
    )
    assert clear_index < upload_index
    assert old_path not in workspace.backend.files
    assert (["git", "init", "--quiet"], case.runtime_workspace, case.command_timeout) in workspace.backend.commands


@pytest.mark.asyncio
async def test_workspace_upload_rejects_unsafe_host_archive(tmp_path):
    """Host archives cannot use traversal paths to write outside the workspace."""
    source = tmp_path / "unsafe.tar.gz"
    with tarfile.open(source, mode="w:gz") as archive:
        info = tarfile.TarInfo("../outside.md")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"nope"))
    workspace = FakeWorkspace("query-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("query-1")
    commands_before = list(workspace.backend.commands)

    with pytest.raises(ValueError, match="unsafe path"):
        await case.upload_workspace(source)

    assert workspace.backend.commands == commands_before


@pytest.mark.asyncio
async def test_workspace_upload_accepts_a_complete_case_export(tmp_path):
    """A previous ``export()`` archive restores only its runtime workspace."""
    source = tmp_path / "case-export.tar.gz"
    with tarfile.open(source, mode="w:gz") as archive:
        manifest = b'{"case_id": "source"}'
        manifest_info = tarfile.TarInfo("manifest.json")
        manifest_info.size = len(manifest)
        archive.addfile(manifest_info, io.BytesIO(manifest))
        note = b"remember this\n"
        note_info = tarfile.TarInfo("reme_workspace/daily/2023-05-30.md")
        note_info.size = len(note)
        archive.addfile(note_info, io.BytesIO(note))
        log = b"do not import case logs"
        log_info = tarfile.TarInfo("logs/actions.jsonl")
        log_info.size = len(log)
        archive.addfile(log_info, io.BytesIO(log))
    workspace = FakeWorkspace("query-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("query-1")

    await case.upload_workspace(source)

    payload = workspace.backend.files["/tmp/reme-workspace-upload.tar.gz"]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert archive.getnames() == ["daily/2023-05-30.md"]
        assert archive.extractfile("daily/2023-05-30.md").read() == note


@pytest.mark.asyncio
async def test_host_selects_when_to_commit_configured_daily_directory():
    """Several ingested sessions can be grouped into one explicit host checkpoint."""
    workspace = FakeWorkspace("case-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")

    first = await case.ingest_session(messages=[], session_id="session-001", update_index=False)
    second = await case.ingest_session(messages=[], session_id="session-002", update_index=False)

    assert first.success is second.success is True
    commands = workspace.backend.commands
    assert not any("commit" in command for command, _, _ in commands)

    await case.commit_memory_history("sessions: session-001, session-002")

    assert (["git", "init", "--quiet"], case.runtime_workspace, case.command_timeout) in commands
    assert (["git", "add", "-A", "--", "daily"], case.runtime_workspace, case.command_timeout) in commands
    commit = next(command for command, cwd, _ in commands if "commit" in command and cwd == case.runtime_workspace)
    assert "--allow-empty" in commit
    assert "--only" in commit
    assert commit[-2:] == ["-m", "sessions: session-001, session-002"]
    assert all(path not in commit for path in ("metadata", "session", "resource", "."))


def test_worker_memory_checkpoint_records_only_daily_source_files(tmp_path):
    """Build checkpoints expose a session boundary without committing derived state."""

    runtime_workspace = tmp_path / "reme_workspace"
    daily = runtime_workspace / "daily"
    metadata = runtime_workspace / "metadata"
    daily.mkdir(parents=True)
    metadata.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=runtime_workspace, check=True)
    (daily / "memory.md").write_text("remember\n", encoding="utf-8")
    (metadata / "index.json").write_text("{}\n", encoding="utf-8")

    checkpoint = worker._commit_memory_checkpoint(  # pylint: disable=protected-access
        tmp_path,
        {"workspace_dir": str(runtime_workspace), "daily_dir": "daily"},
        "session: one",
    )

    assert checkpoint["message"] == "session: one"
    assert checkpoint["path"] == "daily"
    assert (
        subprocess.run(
            ["git", "show", "--pretty=", "--name-only", checkpoint["commit_sha"]],
            cwd=runtime_workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        == "daily/memory.md"
    )
    assert (
        subprocess.run(
            ["git", "status", "--short"],
            cwd=runtime_workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        == "?? metadata/"
    )


@pytest.mark.asyncio
async def test_memory_history_uses_resolved_custom_daily_directory():
    """Git pathspecs follow the candidate's validated daily_dir configuration."""
    workspace = FakeWorkspace("case-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")
    workspace.backend.files["/workspace/case/runtime-layout.json"] = json.dumps(
        {
            "workspace_root": "reme_workspace",
            "configured_paths": {"daily_dir": "reme_workspace/memories/daily-notes"},
            "analysis_excludes": [],
        },
    ).encode()

    assert await case._daily_history_path() == "memories/daily-notes"  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_failed_ingest_session_does_not_create_history_commit():
    """Failed memory construction must not be represented as a successful session boundary."""

    class FailingJobBackend(FakeBackend):
        """Return a failed response for every emulated ReMe job."""

        async def exec_shell(self, command, *, cwd=None, timeout=None):
            result = await super().exec_shell(command, cwd=cwd, timeout=timeout)
            if "/workspace/harness/worker.py" in command:
                response_path = command[command.index("--response") + 1]
                self.files[response_path] = json.dumps(
                    {"success": False, "answer": "failed", "metadata": {}, "error": "failed"},
                ).encode()
            return result

    backend = FailingJobBackend()
    workspace = FakeWorkspace("case-1", "candidate:test", {}, backend=backend)
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")

    result = await case.ingest_session(messages=[], session_id="session-001", update_index=False)

    assert result.success is False
    assert not any("commit" in command for command, _, _ in backend.commands)


@pytest.mark.asyncio
async def test_one_container_can_reset_and_run_another_case(tmp_path):
    """Reset removes case state while retaining the installed candidate and worker."""
    workspaces = []

    def build(case_id, image, env):
        """Capture the single workspace reused across cases."""
        workspace = FakeWorkspace(case_id, image, env)
        workspaces.append(workspace)
        return workspace

    factory = DockerReMeSandboxFactory(ImageCandidate("reme-candidate:test"), workspace_builder=build)
    case = await factory.create_case("case-1")
    await case.run_job("auto_memory", {"session_id": "first"})
    await case.export(tmp_path / "case-1.tar.gz")

    backend = workspaces[0].backend
    worker_bytes = backend.files["/workspace/harness/worker.py"]
    backend.files["/workspace/candidate/reme/__init__.py"] = b"candidate-code"
    old_git_object = "/workspace/case/reme_workspace/.git/objects/old-case-object"
    backend.files[old_git_object] = b"old case history"
    backend.files["/tmp/reme-workspace-upload.tar.gz"] = b"uploaded workspace"
    backend.files["/tmp/reme-workspace-export.tar.gz"] = b"exported workspace"
    assert any(path.startswith("/workspace/case/") for path in backend.files)

    await case.reset_case("case-2")

    assert case.case_id == "case-2"
    assert len(workspaces) == 1
    assert backend.files["/workspace/harness/worker.py"] == worker_bytes
    assert backend.files["/workspace/candidate/reme/__init__.py"] == b"candidate-code"
    assert old_git_object not in backend.files
    assert backend.files["/workspace/case/reme_workspace/.git/HEAD"].startswith(b"ref:")
    assert not any(path.startswith("/workspace/case/results/") for path in backend.files)
    assert "/tmp/reme-case-export.tar.gz" not in backend.files
    assert "/tmp/reme-workspace-upload.tar.gz" not in backend.files
    assert "/tmp/reme-workspace-export.tar.gz" not in backend.files
    git_init_positions = [
        index for index, (command, _, _) in enumerate(backend.commands) if command == ["git", "init", "--quiet"]
    ]
    clear_position = next(
        index for index, (command, _, _) in enumerate(backend.commands) if command[:2] == ["rm", "-rf"]
    )
    assert len(git_init_positions) == 2
    assert git_init_positions[0] < clear_position < git_init_positions[1]

    result = await case.run_job("agentic_answer", {"query": "second"})
    await case.export(tmp_path / "case-2.tar.gz")

    assert result.success is True
    manifest = json.loads(backend.files["/workspace/case/manifest.json"])
    assert manifest["case_id"] == "case-2"


@pytest.mark.asyncio
async def test_full_export_keeps_complete_case_tree(tmp_path):
    """The explicit full profile preserves the previous export contract."""
    workspace = FakeWorkspace("case-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")

    await case.export_full(tmp_path / "full.tar.gz")

    export_command = next(command for command, _, _ in workspace.backend.commands if command[:2] == ["tar", "-czf"])
    assert export_command == [
        "tar",
        "-czf",
        "/tmp/reme-case-export.tar.gz",
        "-C",
        "/workspace/case",
        ".",
    ]
    manifest = json.loads(workspace.backend.files["/workspace/case/manifest.json"])
    assert manifest["export_profile"] == "full"


@pytest.mark.asyncio
async def test_analysis_export_requires_runtime_layout(tmp_path):
    """Selective export fails clearly before any job resolves configured paths."""
    workspace = FakeWorkspace("case-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")

    with pytest.raises(RuntimeError, match="completed job"):
        await case.export(tmp_path / "analysis.tar.gz")


def test_runtime_layout_uses_resolved_custom_memory_directories(tmp_path):
    """Selective export paths come from config instead of fixed directory names."""
    workspace = tmp_path / "runtime"
    config = ApplicationConfig(
        workspace_dir=str(workspace),
        daily_dir="memories/daily-notes",
        digest_dir="memories/digests",
        mem_session_dir="traces/agent-sessions",
    )

    worker._write_runtime_layout(tmp_path, config)  # pylint: disable=protected-access

    layout = json.loads((tmp_path / "runtime-layout.json").read_text())
    assert layout["configured_paths"]["daily_dir"] == "runtime/memories/daily-notes"
    assert layout["configured_paths"]["digest_dir"] == "runtime/memories/digests"
    assert layout["configured_paths"]["mem_session_dir"] == "runtime/traces/agent-sessions"
    assert layout["analysis_excludes"] == [
        "runtime/metadata",
        "runtime/resource",
    ]


@pytest.mark.asyncio
async def test_reset_case_rejects_unsafe_id_without_deleting_state():
    """An invalid next case ID cannot trigger cleanup."""
    workspace = FakeWorkspace("case-1", "candidate:test", {})
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")
    before = list(workspace.backend.commands)

    with pytest.raises(ValueError, match="case_id"):
        await case.reset_case("../case-2")

    assert case.case_id == "case-1"
    assert workspace.backend.commands == before


@pytest.mark.asyncio
async def test_jobs_in_one_container_are_serialized():
    """Concurrent host calls cannot execute two case jobs at once."""

    class TrackingBackend(FakeBackend):
        """Measure concurrent worker processes in one fake container."""

        def __init__(self):
            super().__init__()
            self.active_jobs = 0
            self.max_active_jobs = 0

        async def exec_shell(self, command, *, cwd=None, timeout=None):
            is_job = "/workspace/harness/worker.py" in command
            if is_job:
                self.active_jobs += 1
                self.max_active_jobs = max(self.max_active_jobs, self.active_jobs)
                await asyncio.sleep(0.01)
            try:
                return await super().exec_shell(command, cwd=cwd, timeout=timeout)
            finally:
                if is_job:
                    self.active_jobs -= 1

    backend = TrackingBackend()
    workspace = FakeWorkspace("case-1", "candidate:test", {}, backend=backend)
    factory = DockerReMeSandboxFactory(
        ImageCandidate("candidate:test"),
        workspace_builder=lambda *_args: workspace,
    )
    case = await factory.create_case("case-1")

    first, second = await asyncio.gather(
        case.run_job("agentic_answer", {"query": "first"}),
        case.run_job("agentic_answer", {"query": "second"}),
    )

    assert first.success is second.success is True
    assert backend.max_active_jobs == 1


@pytest.mark.asyncio
async def test_worker_calls_application_directly_and_always_closes(monkeypatch, tmp_path):
    """The worker calls Application.run_job without any HTTP service."""
    events = []
    monkeypatch.setenv("TMPDIR", str(tmp_path / "original-tmp"))

    class FakeReMe:
        """Record the direct ReMe application lifecycle."""

        def __init__(self, **config):
            """Record construction."""
            events.append(("construct", config["workspace_dir"], config["daily_dir"]))
            self.context = SimpleNamespace(
                metadata={},
                components={ComponentEnum.AGENT_WRAPPER: {"bench": SimpleNamespace()}},
                jobs={"agentic_answer": SimpleNamespace(), "search": SimpleNamespace()},
            )

        async def start(self):
            """Record startup."""
            events.append("start")

        async def run_job(self, name, **arguments):
            """Return a successful direct job response."""
            events.append(("run_job", name, arguments))
            global_counter_add(self.context.metadata, ["__token_counter", "bench", "input_tokens"], 12)
            global_counter_add(self.context.metadata, ["__token_counter", "bench", "output_tokens"], 3)
            global_counter_add(self.context.metadata, ["__token_counter", "bench", "total_tokens"], 15)
            global_counter_add(self.context.metadata, ["__job_counter", "agentic_answer"], 1)
            global_counter_add(self.context.metadata, ["__job_counter", "search"], 3)
            return SimpleNamespace(success=True, answer="answer", metadata={"direct": True})

        async def close(self):
            """Record deterministic shutdown."""
            events.append("close")

    monkeypatch.setattr("reme.config.resolve_app_config", lambda **kwargs: kwargs)
    monkeypatch.setattr("reme.reme.ReMe", FakeReMe)
    result = await worker._run(  # pylint: disable=protected-access
        {
            "job": "agentic_answer",
            "arguments": {"query": "hello"},
            "config": "lme.yaml",
            "case_root": str(tmp_path),
            "workspace_dir": str(tmp_path / "workspace"),
        },
    )

    assert result == {
        "success": True,
        "answer": "answer",
        "metadata": {"direct": True},
        "token_usage": {
            "bench": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
        },
        "job_call_counts": {"agentic_answer": 1, "search": 3},
        "error": None,
    }
    assert worker.os.environ["TMPDIR"] == str(tmp_path / "tmp")
    layout = json.loads((tmp_path / "runtime-layout.json").read_text())
    assert layout["workspace_root"] == "workspace"
    assert layout["analysis_excludes"] == [
        "workspace/metadata",
        "workspace/resource",
    ]
    assert events == [
        ("construct", str(tmp_path / "workspace"), "daily"),
        "start",
        ("run_job", "agentic_answer", {"query": "hello"}),
        "close",
    ]


@pytest.mark.asyncio
async def test_worker_reuses_one_application_per_phase_and_isolates_query_logs(monkeypatch, tmp_path):
    """Construction and all queries use two Applications while every query gets only its own log."""
    application_ids = []

    class FakeReMe:
        """Minimal multi-job Application whose log messages expose phase leakage."""

        def __init__(self, **_config):
            self.application_id = len(application_ids) + 1
            application_ids.append(self.application_id)
            self.context = SimpleNamespace(
                metadata={},
                components={},
                jobs={
                    "auto_memory": SimpleNamespace(),
                    "index_update": SimpleNamespace(),
                    "agentic_answer": SimpleNamespace(),
                    "answer_judge": SimpleNamespace(),
                },
            )

        async def start(self):
            """Start without external resources."""

        async def run_job(self, name, **arguments):
            """Log a unique marker and return deterministic answer/judge results."""
            from reme.utils import get_logger

            marker = arguments.get("query", arguments.get("session_id", "judge"))
            get_logger().info(f"job={name} marker={marker}")
            answer = "yes" if name == "answer_judge" else f"answer:{marker}"
            return SimpleNamespace(success=True, answer=answer, metadata={"job": name})

        async def close(self):
            """Close without external resources."""

    monkeypatch.setattr("reme.config.resolve_app_config", lambda **kwargs: kwargs)
    monkeypatch.setattr("reme.reme.ReMe", FakeReMe)
    common = {
        "config": "lme.yaml",
        "case_id": "case-1",
        "case_root": str(tmp_path),
        "workspace_dir": str(tmp_path / "workspace"),
    }

    build = await worker._run_build(  # pylint: disable=protected-access
        {
            **common,
            "jobs": [
                {"job": "auto_memory", "arguments": {"session_id": "construction-only"}},
                {"job": "index_update", "arguments": {}},
            ],
        },
    )
    queries = await worker._run_queries(  # pylint: disable=protected-access
        {
            **common,
            "queries": [
                {
                    "query_id": "query:1",
                    "question": "first-only",
                    "golden_answer": "a",
                    "judge_arguments": {},
                    "score_mapping": {"yes": 1.0, "no": 0.0},
                },
                {
                    "query_id": "query:2",
                    "question": "second-only",
                    "golden_answer": "b",
                    "judge_arguments": {},
                    "score_mapping": {"yes": 1.0, "no": 0.0},
                },
            ],
        },
    )

    assert build["success"] is True
    assert queries["success"] is True
    assert application_ids == [1, 2]
    build_log = (tmp_path / "build_log/build.log").read_text(encoding="utf-8")
    first_log = (tmp_path / "queries/query:1/answer.log").read_text(encoding="utf-8")
    second_log = (tmp_path / "queries/query:2/answer.log").read_text(encoding="utf-8")
    assert "construction-only" in build_log
    assert "first-only" not in build_log
    assert "first-only" in first_log
    assert "second-only" not in first_log
    assert "construction-only" not in first_log
    assert "second-only" in second_log
    assert "first-only" not in second_log

    summary = json.loads((tmp_path / "queries/summary.json").read_text(encoding="utf-8"))
    assert summary["queries"] == [
        {"query_id": "query:1", "score": 1.0},
        {"query_id": "query:2", "score": 1.0},
    ]
    assert all("directory" not in item for item in summary["queries"])


@pytest.mark.asyncio
async def test_worker_returns_usage_accumulated_before_job_failure(monkeypatch, tmp_path):
    """A failed step still reports tokens spent before the exception."""

    class FailingReMe:
        """Record token usage and then fail the requested job."""

        def __init__(self, **_config):
            self.context = SimpleNamespace(
                metadata={},
                components={ComponentEnum.AGENT_WRAPPER: {"bench": SimpleNamespace()}},
                jobs={"agentic_answer": SimpleNamespace()},
            )

        async def start(self):
            """Start without side effects."""

        async def run_job(self, _name, **_arguments):
            """Consume tokens before surfacing a step error."""
            global_counter_add(self.context.metadata, ["__token_counter", "bench", "input_tokens"], 8)
            global_counter_add(self.context.metadata, ["__token_counter", "bench", "output_tokens"], 2)
            global_counter_add(self.context.metadata, ["__token_counter", "bench", "total_tokens"], 10)
            global_counter_add(self.context.metadata, ["__job_counter", "agentic_answer"], 1)
            raise RuntimeError("step failed")

        async def close(self):
            """Close without side effects."""

    monkeypatch.setattr("reme.config.resolve_app_config", lambda **kwargs: kwargs)
    monkeypatch.setattr("reme.reme.ReMe", FailingReMe)

    result = await worker._run(  # pylint: disable=protected-access
        {
            "job": "agentic_answer",
            "config": "lme.yaml",
            "case_root": str(tmp_path),
            "workspace_dir": str(tmp_path / "workspace"),
        },
    )

    assert result["success"] is False
    assert result["error"] == "RuntimeError: step failed"
    assert result["token_usage"] == {
        "bench": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
    }
    assert result["job_call_counts"] == {"agentic_answer": 1}
