"""AgentScope DockerWorkspace orchestration for isolated ReMe cases."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable, Protocol
import uuid

from .candidate import ImageCandidate, SourceCandidate
from .models import ActionRecord, JobResult

CASE_ROOT = "/workspace/case"
CANDIDATE_ROOT = "/workspace/candidate"
CANDIDATE_VENV = "/workspace/candidate-venv"
HARNESS_ROOT = "/workspace/harness"
SOURCE_ARCHIVE = "/workspace/candidate.tar.gz"
EXPORT_ARCHIVE = "/tmp/reme-case-export.tar.gz"
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class Backend(Protocol):
    """Subset of AgentScope's backend API used by this module."""

    async def exec_shell(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Execute an argv vector in the sandbox."""
        raise NotImplementedError

    async def read_file(self, path: str) -> bytes:
        """Read a sandbox file."""
        raise NotImplementedError

    async def write_file(self, path: str, data: bytes) -> None:
        """Write a sandbox file."""
        raise NotImplementedError


class Workspace(Protocol):
    """Subset of AgentScope's DockerWorkspace used by this module."""

    async def initialize(self) -> None:
        """Start the workspace."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the workspace."""
        raise NotImplementedError

    def get_backend(self) -> Backend:
        """Return the execution backend."""
        raise NotImplementedError


WorkspaceBuilder = Callable[[str, str, dict[str, str]], Workspace]


class SandboxCommandError(RuntimeError):
    """A required command failed inside a case sandbox."""


def _utc_now() -> str:
    """Return an audit-friendly UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _decode(value: bytes | str | None) -> str:
    """Decode backend process output without losing diagnostics."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _default_workspace_builder(case_id: str, image: str, env: dict[str, str]) -> Workspace:
    """Create the exact AgentScope 2.0.4.post1 Docker backend lazily."""
    import agentscope

    from .direct_docker_workspace import DirectDockerWorkspace

    if agentscope.__version__ != "2.0.4.post1":
        raise RuntimeError(f"sandbox host requires agentscope 2.0.4.post1, got {agentscope.__version__}")
    return DirectDockerWorkspace(
        workspace_id=f"reme-benchmark-{case_id}",
        base_image=image,
        env=env,
    )


class DockerReMeSandboxFactory:
    """Create independent case sandboxes that share one candidate definition.

    A :class:`SourceCandidate` stores one immutable source archive in host
    memory. The same bytes are uploaded to every case, while each case gets a
    new Docker container and a fresh ReMe runtime workspace. An
    :class:`ImageCandidate` skips source upload and uses an image in which ReMe
    is already installed.
    """

    def __init__(
        self,
        candidate: SourceCandidate | ImageCandidate,
        *,
        env: dict[str, str] | None = None,
        config: str = "lme.yaml",
        command_timeout: float = 1800.0,
        workspace_builder: WorkspaceBuilder | None = None,
    ) -> None:
        self.candidate = candidate
        self.env = dict(env or {})
        self.config = config
        self.command_timeout = command_timeout
        self._workspace_builder = workspace_builder or _default_workspace_builder

    async def create_case(self, case_id: str) -> "DockerReMeSandbox":
        """Start and prepare a new container for one benchmark case."""
        if not _CASE_ID_RE.fullmatch(case_id):
            raise ValueError("case_id must contain only letters, numbers, '.', '_' or '-' and be <= 128 characters")
        image = self.candidate.base_image if isinstance(self.candidate, SourceCandidate) else self.candidate.image
        workspace = self._workspace_builder(case_id, image, self.env)
        case = DockerReMeSandbox(
            case_id=case_id,
            candidate=self.candidate,
            workspace=workspace,
            config=self.config,
            command_timeout=self.command_timeout,
            environment_names=sorted(self.env),
        )
        try:
            await case.initialize()
        except Exception:
            await workspace.close()
            raise
        return case

    async def create_cases(self, case_ids: list[str], *, concurrency: int = 4) -> list["DockerReMeSandbox"]:
        """Create several independent cases while bounding Docker pressure."""
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_ids must be unique")
        semaphore = asyncio.Semaphore(concurrency)

        async def create(case_id: str) -> DockerReMeSandbox:
            async with semaphore:
                return await self.create_case(case_id)

        results = await asyncio.gather(*(create(case_id) for case_id in case_ids), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            await asyncio.gather(
                *(result.close() for result in results if isinstance(result, DockerReMeSandbox)),
            )
            raise failures[0]
        return [result for result in results if isinstance(result, DockerReMeSandbox)]


class DockerReMeSandbox:
    """One isolated Docker container and one fresh ReMe runtime workspace."""

    def __init__(
        self,
        *,
        case_id: str,
        candidate: SourceCandidate | ImageCandidate,
        workspace: Workspace,
        config: str,
        command_timeout: float,
        environment_names: list[str],
    ) -> None:
        self.case_id = case_id
        self.candidate = candidate
        self.workspace = workspace
        self.config = config
        self.command_timeout = command_timeout
        self.environment_names = environment_names
        self.backend: Backend | None = None
        self.python = "python3"
        self.actions: list[ActionRecord] = []
        self._closed = False

    @property
    def runtime_workspace(self) -> str:
        """ReMe-owned runtime workspace inside this case container."""
        return f"{CASE_ROOT}/reme_workspace"

    @property
    def logs_dir(self) -> str:
        """Directory containing ReMe and orchestration logs."""
        return f"{CASE_ROOT}/logs"

    @property
    def results_dir(self) -> str:
        """Directory containing job, answer and score artifacts."""
        return f"{CASE_ROOT}/results"

    async def initialize(self) -> None:
        """Start Docker, upload the worker, and prepare the selected candidate."""
        await self.workspace.initialize()
        self.backend = self.workspace.get_backend()
        await self._exec_checked(
            "create_case_layout",
            [
                "mkdir",
                "-p",
                f"{CASE_ROOT}/inbox",
                self.runtime_workspace,
                self.logs_dir,
                self.results_dir,
                HARNESS_ROOT,
            ],
        )
        worker = Path(__file__).with_name("worker.py").read_bytes()
        await self.backend.write_file(f"{HARNESS_ROOT}/worker.py", worker)

        if isinstance(self.candidate, SourceCandidate):
            await self._install_source_candidate()
        else:
            await self._validate_candidate(self.python)

    async def _install_source_candidate(self) -> None:
        assert self.backend is not None
        await self.backend.write_file(SOURCE_ARCHIVE, self.candidate.snapshot.archive)
        await self._exec_checked("create_candidate_dir", ["mkdir", "-p", CANDIDATE_ROOT])
        await self._exec_checked(
            "extract_candidate",
            ["tar", "-xzf", SOURCE_ARCHIVE, "-C", CANDIDATE_ROOT],
        )
        await self._exec_checked(
            "create_candidate_venv",
            ["python3", "-m", "venv", "--system-site-packages", CANDIDATE_VENV],
        )
        self.python = f"{CANDIDATE_VENV}/bin/python"
        await self._exec_checked(
            "install_candidate",
            [
                self.python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--editable",
                CANDIDATE_ROOT,
            ],
        )
        await self._validate_candidate(self.python)

    async def _validate_candidate(self, python: str) -> None:
        script = (
            "import agentscope, json, reme; "
            "print(json.dumps({'reme': reme.__file__, 'agentscope': agentscope.__version__}))"
        )
        result = await self._exec_checked("validate_candidate", [python, "-c", script])
        validation = json.loads(_decode(result.stdout))
        if validation.get("agentscope") != "2.0.4.post1":
            raise SandboxCommandError(
                f"candidate environment requires agentscope 2.0.4.post1, got {validation.get('agentscope')!r}",
            )
        if isinstance(self.candidate, SourceCandidate) and CANDIDATE_ROOT not in str(validation.get("reme")):
            raise SandboxCommandError(
                f"candidate import did not resolve under {CANDIDATE_ROOT}: {validation.get('reme')}",
            )

    async def _exec(self, name: str, command: list[str], *, cwd: str | None = None) -> Any:
        assert self.backend is not None, "sandbox is not initialized"
        started = _utc_now()
        result = await self.backend.exec_shell(command, cwd=cwd, timeout=self.command_timeout)
        self.actions.append(
            ActionRecord(
                name=name,
                exit_code=int(result.exit_code),
                stdout=_decode(result.stdout),
                stderr=_decode(result.stderr),
                started_at=started,
                finished_at=_utc_now(),
            ),
        )
        return result

    async def _exec_checked(self, name: str, command: list[str], *, cwd: str | None = None) -> Any:
        result = await self._exec(name, command, cwd=cwd)
        if not result.ok():
            raise SandboxCommandError(
                f"sandbox action {name!r} failed with exit {result.exit_code}: {_decode(result.stderr)}",
            )
        return result

    async def run_job(self, job: str, arguments: dict[str, Any] | None = None) -> JobResult:
        """Run one ReMe job directly in-process inside the container."""
        assert self.backend is not None, "sandbox is not initialized"
        request_id = uuid.uuid4().hex
        request_path = f"{CASE_ROOT}/inbox/{request_id}.json"
        response_path = f"{self.results_dir}/{request_id}.json"
        request = {
            "job": job,
            "arguments": arguments or {},
            "config": self.config,
            "case_root": CASE_ROOT,
            "workspace_dir": self.runtime_workspace,
        }
        await self.backend.write_file(request_path, json.dumps(request, ensure_ascii=False).encode("utf-8"))
        await self._exec(
            f"job:{job}",
            [self.python, f"{HARNESS_ROOT}/worker.py", "--request", request_path, "--response", response_path],
            cwd=CASE_ROOT,
        )
        try:
            payload = json.loads((await self.backend.read_file(response_path)).decode("utf-8"))
        except FileNotFoundError as exc:
            raise SandboxCommandError(f"job {job!r} did not produce {response_path}") from exc
        return JobResult.from_dict(payload)

    async def ingest_session(
        self,
        *,
        messages: list[dict[str, Any]],
        session_id: str,
        date: str = "",
        memory_hint: str = "",
        update_index: bool = True,
    ) -> JobResult:
        """Add one session, optionally making it searchable immediately."""
        result = await self.run_job(
            "auto_memory",
            {"messages": messages, "session_id": session_id, "date": date, "memory_hint": memory_hint},
        )
        if result.success and update_index:
            index_result = await self.run_job("index_update")
            if not index_result.success:
                return index_result
        return result

    async def answer(self, *, query: str, query_time: str = "") -> JobResult:
        """Generate an answer with the candidate's agentic_answer job."""
        result = await self.run_job("agentic_answer", {"query": query, "query_time": query_time})
        await self._write_named_result("answer.json", result)
        return result

    async def judge(
        self,
        *,
        query: str,
        agent_answer: str,
        golden_answer: str,
        question_type: str = "",
    ) -> JobResult:
        """Run LLM-as-Judge and persist a normalized numeric score."""
        result = await self.run_job(
            "answer_judge",
            {
                "query": query,
                "agent_answer": agent_answer,
                "golden_answer": golden_answer,
                "question_type": question_type,
            },
        )
        verdict = str(result.answer).strip().lower()
        score = 1 if verdict == "yes" else 0 if verdict == "no" else None
        assert self.backend is not None
        await self.backend.write_file(
            f"{self.results_dir}/score.json",
            json.dumps(
                {"success": result.success, "score": score, "verdict": verdict, "raw": asdict(result)},
                ensure_ascii=False,
                default=str,
            ).encode("utf-8"),
        )
        return result

    async def _write_named_result(self, name: str, result: JobResult) -> None:
        assert self.backend is not None
        await self.backend.write_file(
            f"{self.results_dir}/{name}",
            json.dumps(asdict(result), ensure_ascii=False, default=str).encode("utf-8"),
        )

    async def export(self, destination: str | Path, *, include_candidate: bool = False) -> Path:
        """Download the complete runtime workspace, logs, results and manifest."""
        assert self.backend is not None, "sandbox is not initialized"
        if include_candidate and isinstance(self.candidate, ImageCandidate):
            raise ValueError("an image candidate has no source directory to export")
        candidate_id = (
            self.candidate.candidate_id
            if isinstance(self.candidate, SourceCandidate)
            else self.candidate.resolved_candidate_id
        )
        manifest = {
            "case_id": self.case_id,
            "candidate_id": candidate_id,
            "candidate_mode": "source" if isinstance(self.candidate, SourceCandidate) else "image",
            "config": self.config,
            "environment_names": self.environment_names,
            "exported_at": _utc_now(),
        }
        await self.backend.write_file(
            f"{CASE_ROOT}/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        await self.backend.write_file(
            f"{self.logs_dir}/actions.jsonl",
            b"".join(
                (json.dumps(asdict(action), ensure_ascii=False) + "\n").encode("utf-8") for action in self.actions
            ),
        )
        command = ["tar", "-czf", EXPORT_ARCHIVE, "-C", CASE_ROOT, "."]
        if include_candidate:
            command.extend(["-C", "/workspace", "candidate"])
        await self._exec_checked("export_case", command)
        payload = await self.backend.read_file(EXPORT_ARCHIVE)
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    async def close(self) -> None:
        """Destroy the case container. Export before calling this method."""
        if not self._closed:
            self._closed = True
            await self.workspace.close()

    async def __aenter__(self) -> "DockerReMeSandbox":
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()
