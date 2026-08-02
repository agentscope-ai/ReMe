"""Tests for the standalone Docker benchmark sandbox infrastructure."""

from __future__ import annotations

import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from sandbox import DockerReMeSandboxFactory, ImageCandidate, SourceCandidate, SourceSnapshot
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
        if command[:2] == ["python3", "-c"] or (len(command) > 2 and command[1:3] == ["-c", command[2]]):
            stdout = b'{"reme": "/workspace/candidate/reme/__init__.py", "agentscope": "2.0.4.post1"}\n'
            return FakeExecResult(stdout=stdout)
        if "/workspace/harness/worker.py" in command:
            response_path = command[command.index("--response") + 1]
            self.files[response_path] = json.dumps(
                {"success": True, "answer": "yes", "metadata": {"job": "ok"}, "error": None},
            ).encode()
        if command[:2] == ["tar", "-czf"]:
            self.files[command[2]] = b"case-archive"
        return FakeExecResult()


class FakeWorkspace:
    """A distinct workspace object for each factory call."""

    def __init__(self, case_id, image, env):
        """Initialize one unique fake workspace."""
        self.case_id = case_id
        self.image = image
        self.env = env
        self.backend = FakeBackend()
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
    assert "/workspace/candidate.tar.gz" not in workspaces[0].backend.files
    assert exported.read_bytes() == b"case-archive"
    assert json.loads(workspaces[0].backend.files["/workspace/case/manifest.json"])["candidate_mode"] == "image"


@pytest.mark.asyncio
async def test_worker_calls_application_directly_and_always_closes(monkeypatch, tmp_path):
    """The worker calls Application.run_job without any HTTP service."""
    events = []

    class FakeReMe:
        """Record the direct ReMe application lifecycle."""

        def __init__(self, **config):
            """Record construction."""
            events.append(("construct", config["workspace_dir"]))

        async def start(self):
            """Record startup."""
            events.append("start")

        async def run_job(self, name, **arguments):
            """Return a successful direct job response."""
            events.append(("run_job", name, arguments))
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

    assert result == {"success": True, "answer": "answer", "metadata": {"direct": True}, "error": None}
    assert events == [
        ("construct", str(tmp_path / "workspace")),
        "start",
        ("run_job", "agentic_answer", {"query": "hello"}),
        "close",
    ]
