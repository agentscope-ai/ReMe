"""AgentScope DockerWorkspace orchestration for isolated ReMe cases."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import tarfile
from typing import Any, Callable, Literal, Protocol
import uuid

from .candidate import ImageCandidate, SourceCandidate
from .models import ActionRecord, EvaluationQuery, JobRequest, JobResult

CASE_ROOT = "/workspace/case"
CANDIDATE_ROOT = "/workspace/candidate"
CANDIDATE_VENV = "/workspace/candidate-venv"
HARNESS_ROOT = "/workspace/harness"
SOURCE_ARCHIVE = "/workspace/candidate.tar.gz"
EXPORT_ARCHIVE = "/tmp/reme-case-export.tar.gz"
WORKSPACE_ARCHIVE = "/tmp/reme-workspace-upload.tar.gz"
WORKSPACE_EXPORT_ARCHIVE = "/tmp/reme-workspace-export.tar.gz"
RUNTIME_LAYOUT = f"{CASE_ROOT}/runtime-layout.json"
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_GIT_AUTHOR_NAME = "ReMe Sandbox"
_GIT_AUTHOR_EMAIL = "reme-sandbox@localhost"


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


def _validate_case_id(case_id: str) -> None:
    """Reject identifiers that are unsafe for Docker names and manifests."""
    if not _CASE_ID_RE.fullmatch(case_id):
        raise ValueError("case_id must contain only letters, numbers, '.', '_' or '-' and be <= 128 characters")


def _utc_now() -> str:
    """Return an audit-friendly UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _decode(value: bytes | str | None) -> str:
    """Decode backend process output without losing diagnostics."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _workspace_member_path(name: str) -> PurePosixPath | None:
    """Validate and normalize one relative path in a workspace archive."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"workspace archive contains unsafe path: {name!r}")
    if path == PurePosixPath("."):
        return None
    return path


def _archive_workspace(source: str | Path) -> bytes:
    """Create a safe, deterministic gzip tar archive from a host workspace.

    ``source`` may be a workspace directory or an archive previously returned
    by :meth:`DockerReMeSandbox.export_workspace`. Incoming archives are read
    and rebuilt so links, special files, and traversal paths never reach the
    sandbox extractor.
    """
    supplied_path = Path(source).expanduser()
    if supplied_path.is_symlink():
        raise ValueError("workspace source must not be a symbolic link")
    source_path = supplied_path.resolve(strict=True)
    entries: list[tuple[PurePosixPath, tarfile.TarInfo, bytes | None]] = []
    if source_path.is_dir():
        for item in sorted(source_path.rglob("*"), key=lambda value: value.relative_to(source_path).as_posix()):
            relative = item.relative_to(source_path)
            if item.is_symlink():
                raise ValueError(f"workspace source contains a symbolic link: {relative}")
            path = _workspace_member_path(relative.as_posix())
            assert path is not None
            if item.is_dir():
                info = tarfile.TarInfo(path.as_posix())
                info.type = tarfile.DIRTYPE
                info.mode = item.stat().st_mode & 0o777
                entries.append((path, info, None))
            elif item.is_file():
                info = tarfile.TarInfo(path.as_posix())
                info.size = item.stat().st_size
                info.mode = item.stat().st_mode & 0o777
                entries.append((path, info, item.read_bytes()))
            else:
                raise ValueError(f"workspace source contains unsupported file: {relative}")
    elif source_path.is_file():
        with tarfile.open(source_path, mode="r:*") as archive:
            members = archive.getmembers()
            normalized_paths = [path for member in members if (path := _workspace_member_path(member.name)) is not None]
            # ``export()`` archives the whole case, while ``export_workspace``
            # archives only the workspace contents. Accept both forms so an
            # existing case artifact can seed independent query sandboxes.
            case_export = PurePosixPath("manifest.json") in normalized_paths and any(
                path.parts and path.parts[0] == "reme_workspace" for path in normalized_paths
            )
            seen: set[PurePosixPath] = set()
            for member in members:
                path = _workspace_member_path(member.name)
                if path is None:
                    continue
                if case_export:
                    if not path.parts or path.parts[0] != "reme_workspace":
                        continue
                    if len(path.parts) == 1:
                        if not member.isdir():
                            raise ValueError("case export has a non-directory reme_workspace entry")
                        continue
                    path = PurePosixPath(*path.parts[1:])
                if path in seen:
                    raise ValueError(f"workspace archive contains duplicate path: {path}")
                seen.add(path)
                if member.isdir():
                    info = tarfile.TarInfo(path.as_posix())
                    info.type = tarfile.DIRTYPE
                    info.mode = member.mode & 0o777
                    entries.append((path, info, None))
                elif member.isfile():
                    content = archive.extractfile(member)
                    if content is None:
                        raise ValueError(f"workspace archive cannot read file: {path}")
                    data = content.read()
                    info = tarfile.TarInfo(path.as_posix())
                    info.size = len(data)
                    info.mode = member.mode & 0o777
                    entries.append((path, info, data))
                else:
                    raise ValueError(f"workspace archive contains unsupported entry: {member.name!r}")
    else:
        raise ValueError(f"workspace source is neither a directory nor a file: {source_path}")

    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for _path, info, data in entries:
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data) if data is not None else None)
    return raw.getvalue()


def _default_workspace_builder(workspace_id: str, image: str, env: dict[str, str]) -> Workspace:
    """Create the exact AgentScope 2.0.4.post1 Docker backend lazily."""
    import agentscope

    from .direct_docker_workspace import DirectDockerWorkspace

    if agentscope.__version__ != "2.0.4.post1":
        raise RuntimeError(f"sandbox host requires agentscope 2.0.4.post1, got {agentscope.__version__}")
    return DirectDockerWorkspace(
        workspace_id=workspace_id,
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
        self._workspace_prefix = f"reme-benchmark-{uuid.uuid4().hex[:12]}"
        self._workspace_sequence = 0

    def _next_workspace_id(self) -> str:
        """Allocate a factory-scoped container identity independent of case IDs."""
        self._workspace_sequence += 1
        return f"{self._workspace_prefix}-{self._workspace_sequence:04d}"

    async def create_case(self, case_id: str) -> "DockerReMeSandbox":
        """Start and prepare a new container for one benchmark case."""
        _validate_case_id(case_id)
        image = self.candidate.base_image if isinstance(self.candidate, SourceCandidate) else self.candidate.image
        container_id = self._next_workspace_id()
        workspace = self._workspace_builder(container_id, image, self.env)
        case = DockerReMeSandbox(
            case_id=case_id,
            container_id=container_id,
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
        container_id: str,
        candidate: SourceCandidate | ImageCandidate,
        workspace: Workspace,
        config: str,
        command_timeout: float,
        environment_names: list[str],
    ) -> None:
        self.case_id = case_id
        self.container_id = container_id
        self.candidate = candidate
        self.workspace = workspace
        self.config = config
        self.command_timeout = command_timeout
        self.environment_names = environment_names
        self.backend: Backend | None = None
        self.python = "python3"
        self.actions: list[ActionRecord] = []
        self._closed = False
        self._operation_lock = asyncio.Lock()

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
        await self._create_case_layout()
        worker = Path(__file__).with_name("worker.py").read_bytes()
        await self.backend.write_file(f"{HARNESS_ROOT}/worker.py", worker)

        if isinstance(self.candidate, SourceCandidate):
            await self._install_source_candidate()
        else:
            await self._validate_candidate(self.python)

    async def _create_case_layout(self) -> None:
        """Create the disposable directories owned by the active case."""
        await self._exec_checked(
            "create_case_layout",
            [
                "mkdir",
                "-p",
                f"{CASE_ROOT}/inbox",
                f"{CASE_ROOT}/tmp",
                self.runtime_workspace,
                self.logs_dir,
                self.results_dir,
                f"{CASE_ROOT}/build_log",
                f"{CASE_ROOT}/queries",
                HARNESS_ROOT,
            ],
        )
        await self._exec_checked(
            "initialize_memory_history",
            ["git", "init", "--quiet"],
            cwd=self.runtime_workspace,
        )

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

    async def _run_job(self, job: str, arguments: dict[str, Any] | None = None) -> JobResult:
        """Run one ReMe job while the caller holds the operation lock."""
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

    async def _run_worker_request(self, mode: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one structured worker request and return its JSON response."""
        assert self.backend is not None, "sandbox is not initialized"
        request_id = uuid.uuid4().hex
        request_path = f"{CASE_ROOT}/inbox/{request_id}.json"
        response_path = f"{self.results_dir}/{request_id}.json"
        request = {
            "mode": mode,
            "config": self.config,
            "case_id": self.case_id,
            "case_root": CASE_ROOT,
            "workspace_dir": self.runtime_workspace,
            **payload,
        }
        await self.backend.write_file(request_path, json.dumps(request, ensure_ascii=False).encode("utf-8"))
        await self._exec(
            f"worker:{mode}",
            [self.python, f"{HARNESS_ROOT}/worker.py", "--request", request_path, "--response", response_path],
            cwd=CASE_ROOT,
        )
        try:
            result = json.loads((await self.backend.read_file(response_path)).decode("utf-8"))
        except FileNotFoundError as exc:
            raise SandboxCommandError(f"worker mode {mode!r} did not produce {response_path}") from exc
        if not isinstance(result, dict):
            raise SandboxCommandError(f"worker mode {mode!r} returned a non-object response")
        return result

    async def run_job(self, job: str, arguments: dict[str, Any] | None = None) -> JobResult:
        """Run one ReMe job without overlapping another operation."""
        async with self._operation_lock:
            return await self._run_job(job, arguments)

    async def run_build(self, jobs: list[JobRequest | dict[str, Any]]) -> dict[str, Any]:
        """Run all construction jobs in one Application and capture ``build.log``."""
        serialized: list[dict[str, Any]] = []
        for specification in jobs:
            value = asdict(specification) if isinstance(specification, JobRequest) else dict(specification)
            if not isinstance(value.get("job"), str) or not value["job"]:
                raise ValueError("every build job requires a non-empty job name")
            if not isinstance(value.get("arguments", {}), dict):
                raise ValueError("build job arguments must be an object")
            checkpoint = value.get("memory_checkpoint")
            if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
                raise ValueError("build job memory_checkpoint must be a non-empty string or null")
            serialized.append(value)
        if not serialized:
            raise ValueError("run_build requires at least one job")
        async with self._operation_lock:
            return await self._run_worker_request("build", {"jobs": serialized})

    @staticmethod
    def _validate_query_artifact_id(query_id: str) -> None:
        """Keep verbatim query-directory names inside the artifact root."""
        if not query_id or len(query_id.encode("utf-8")) > 255:
            raise ValueError("query_id must be non-empty and no longer than 255 UTF-8 bytes")
        if query_id in {".", ".."} or "/" in query_id or "\\" in query_id or "\x00" in query_id:
            raise ValueError(f"query_id is not a safe directory name: {query_id!r}")
        if query_id == "summary.json":
            raise ValueError("query_id conflicts with the queries summary: 'summary.json'")

    async def run_queries(self, queries: list[EvaluationQuery | dict[str, Any]]) -> dict[str, Any]:
        """Answer and judge multiple queries in one Application with isolated logs."""
        serialized, query_ids = self._serialize_queries(queries)
        if not serialized:
            raise ValueError("run_queries requires at least one query")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query IDs must be unique")
        async with self._operation_lock:
            return await self._run_worker_request("queries", {"queries": serialized})

    async def run_query(self, query: EvaluationQuery | dict[str, Any]) -> dict[str, Any]:
        """Answer and judge one append-only query for a host-side lease."""

        serialized, _ = self._serialize_queries([query])
        async with self._operation_lock:
            response = await self._run_worker_request("query", {"queries": serialized})
        results = response.get("queries")
        if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
            raise SandboxCommandError("single-query worker returned an invalid result")
        return results[0]

    def _serialize_queries(
        self,
        queries: list[EvaluationQuery | dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Validate and serialize query payloads shared by batch and lease APIs."""

        serialized: list[dict[str, Any]] = []
        query_ids: list[str] = []
        for query in queries:
            value = asdict(query) if isinstance(query, EvaluationQuery) else dict(query)
            query_id = value.get("query_id")
            if not isinstance(query_id, str):
                raise ValueError("every evaluation query requires a string query_id")
            self._validate_query_artifact_id(query_id)
            if not isinstance(value.get("question"), str):
                raise ValueError(f"query {query_id!r} requires a string question")
            query_ids.append(query_id)
            serialized.append(value)
        return serialized, query_ids

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
        async with self._operation_lock:
            result = await self._run_job(
                "auto_memory",
                {"messages": messages, "session_id": session_id, "date": date, "memory_hint": memory_hint},
            )
            if result.success and update_index:
                index_result = await self._run_job("index_update")
                if not index_result.success:
                    return index_result
            return result

    async def commit_memory_history(self, message: str) -> None:
        """Create a host-requested checkpoint containing only daily-memory changes."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("memory history commit message must be a non-empty string")
        async with self._operation_lock:
            await self._commit_memory_history(message)

    async def _daily_history_path(self) -> str:
        """Return the validated workspace-relative daily directory from the runtime layout."""
        assert self.backend is not None
        try:
            payload = json.loads((await self.backend.read_file(RUNTIME_LAYOUT)).decode("utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxCommandError(f"cannot resolve daily history path from runtime layout: {exc}") from exc

        expected_root = PurePosixPath(self.runtime_workspace).relative_to(CASE_ROOT)
        configured_paths = payload.get("configured_paths") if isinstance(payload, dict) else None
        if not isinstance(configured_paths, dict):
            raise SandboxCommandError("sandbox runtime layout has invalid configured_paths")
        case_relative = self._validated_workspace_path(
            configured_paths.get("daily_dir"),
            expected_root,
            allow_root=False,
        )
        return PurePosixPath(case_relative).relative_to(expected_root).as_posix()

    async def _commit_memory_history(self, message: str) -> None:
        """Commit one host-selected boundary while tracking only the configured daily directory."""
        daily_path = await self._daily_history_path()
        await self._exec_checked(
            "stage_memory_history",
            ["git", "add", "-A", "--", daily_path],
            cwd=self.runtime_workspace,
        )
        diff_result = await self._exec(
            "check_memory_history",
            ["git", "diff", "--cached", "--quiet", "--", daily_path],
            cwd=self.runtime_workspace,
        )
        if diff_result.exit_code not in (0, 1):
            raise SandboxCommandError(
                f"cannot inspect staged daily-memory changes: {_decode(diff_result.stderr)}",
            )
        commit_command = [
            "git",
            "-c",
            f"user.name={_GIT_AUTHOR_NAME}",
            "-c",
            f"user.email={_GIT_AUTHOR_EMAIL}",
            "commit",
            "--quiet",
            "--allow-empty",
            "--only",
            "-m",
            message,
        ]
        if diff_result.exit_code == 1:
            commit_command.extend(["--", daily_path])
        await self._exec_checked(
            "commit_memory_history",
            commit_command,
            cwd=self.runtime_workspace,
        )

    async def answer(self, *, query: str, query_time: str = "") -> JobResult:
        """Generate an answer with the candidate's agentic_answer job."""
        async with self._operation_lock:
            result = await self._run_job("agentic_answer", {"query": query, "query_time": query_time})
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
        async with self._operation_lock:
            result = await self._run_job(
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

    async def export(
        self,
        destination: str | Path,
        *,
        profile: Literal["analysis", "full"] = "analysis",
        include_candidate: bool = False,
    ) -> Path:
        """Download analysis artifacts, or the full case when requested."""
        if profile not in ("analysis", "full"):
            raise ValueError("export profile must be 'analysis' or 'full'")
        async with self._operation_lock:
            return await self._export(destination, profile=profile, include_candidate=include_candidate)

    async def export_full(self, destination: str | Path, *, include_candidate: bool = False) -> Path:
        """Download every file under the disposable case root."""
        return await self.export(destination, profile="full", include_candidate=include_candidate)

    async def export_workspace(self, destination: str | Path) -> Path:
        """Download a portable snapshot of this runtime workspace to the host.

        The resulting gzip tar archive can be passed directly to
        :meth:`upload_workspace` on any initialized case. It contains the
        workspace contents (including its local Git history), not the enclosing
        case directory, logs, or candidate source.
        """
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            await self._exec_checked(
                "export_workspace",
                ["tar", "-czf", WORKSPACE_EXPORT_ARCHIVE, "-C", self.runtime_workspace, "."],
            )
            target = Path(destination).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(await self.backend.read_file(WORKSPACE_EXPORT_ARCHIVE))
            return target

    async def export_build_log(self, destination: str | Path) -> Path:
        """Download the memory-construction log without wrapping it in an archive."""
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            source = f"{CASE_ROOT}/build_log/build.log"
            try:
                content = await self.backend.read_file(source)
            except FileNotFoundError as exc:
                raise SandboxCommandError(f"build log export is missing required artifact: {source}") from exc
            target = Path(destination).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            return target

    async def export_memory_construction(self, destination: str | Path) -> Path:
        """Download the post-construction workspace and its build log."""
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            required = f"{CASE_ROOT}/build_log/build.log"
            try:
                await self.backend.read_file(required)
            except FileNotFoundError as exc:
                raise SandboxCommandError(
                    f"memory construction export is missing required artifact: {required}",
                ) from exc
            await self._exec_checked(
                "export_memory_construction",
                [
                    "tar",
                    "-czf",
                    EXPORT_ARCHIVE,
                    "-C",
                    CASE_ROOT,
                    "reme_workspace",
                    "build_log",
                ],
            )
            target = Path(destination).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(await self.backend.read_file(EXPORT_ARCHIVE))
            return target

    async def export_queries(self, destination: str | Path) -> Path:
        """Download per-query logs, results, and their aggregate summary."""
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            required = f"{CASE_ROOT}/queries/summary.json"
            try:
                await self.backend.read_file(required)
            except FileNotFoundError as exc:
                raise SandboxCommandError(f"query export is missing required artifact: {required}") from exc
            await self._exec_checked(
                "export_queries",
                ["tar", "-czf", EXPORT_ARCHIVE, "-C", CASE_ROOT, "queries"],
            )
            target = Path(destination).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(await self.backend.read_file(EXPORT_ARCHIVE))
            return target

    async def export_query(self, query_id: str, destination: str | Path) -> Path:
        """Download one leased query's logs and result without a case summary."""

        self._validate_query_artifact_id(query_id)
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            query_root = f"{CASE_ROOT}/queries/{query_id}"
            required = f"{query_root}/result.json"
            try:
                await self.backend.read_file(required)
            except FileNotFoundError as exc:
                raise SandboxCommandError(f"query export is missing required artifact: {required}") from exc
            await self._exec_checked(
                "export_query",
                ["tar", "-czf", EXPORT_ARCHIVE, "-C", CASE_ROOT, f"queries/{query_id}"],
            )
            target = Path(destination).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(await self.backend.read_file(EXPORT_ARCHIVE))
            return target

    async def export_evaluation(self, destination: str | Path) -> Path:
        """Download the workspace, build log, and per-query evaluation artifacts."""
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            for required in (f"{CASE_ROOT}/build_log/build.log", f"{CASE_ROOT}/queries/summary.json"):
                try:
                    await self.backend.read_file(required)
                except FileNotFoundError as exc:
                    raise SandboxCommandError(f"evaluation export is missing required artifact: {required}") from exc
            await self._exec_checked(
                "export_evaluation",
                [
                    "tar",
                    "-czf",
                    EXPORT_ARCHIVE,
                    "-C",
                    CASE_ROOT,
                    "reme_workspace",
                    "build_log",
                    "queries",
                ],
            )
            target = Path(destination).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(await self.backend.read_file(EXPORT_ARCHIVE))
            return target

    async def upload_workspace(self, source: str | Path, *, clear: bool = True) -> None:
        """Copy a host workspace directory or exported archive into this case.

        By default the existing runtime workspace is deleted first, ensuring a
        query case receives only the supplied memory state. Set ``clear=False``
        only when intentionally merging files into the current workspace.
        The copy is made through a validated archive, so each sandbox receives
        an independent filesystem tree and can run queries in parallel.
        """
        archive = _archive_workspace(source)
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            if clear:
                await self._exec_checked("clear_runtime_workspace", ["rm", "-rf", self.runtime_workspace])
            await self._exec_checked("create_runtime_workspace", ["mkdir", "-p", self.runtime_workspace])
            await self.backend.write_file(WORKSPACE_ARCHIVE, archive)
            await self._exec_checked(
                "upload_workspace",
                ["tar", "-xzf", WORKSPACE_ARCHIVE, "-C", self.runtime_workspace],
            )
            # A directory copied from the host may not have Git metadata.  Git
            # init is idempotent and preserves history when the snapshot has it.
            await self._exec_checked(
                "initialize_uploaded_memory_history",
                ["git", "init", "--quiet"],
                cwd=self.runtime_workspace,
            )
            await self._exec_checked("remove_workspace_upload", ["rm", "-f", WORKSPACE_ARCHIVE])

    @staticmethod
    def _validated_workspace_path(value: Any, workspace_root: PurePosixPath, *, allow_root: bool) -> str:
        """Validate one case-relative path received from candidate-controlled state."""
        if not isinstance(value, str):
            raise SandboxCommandError("sandbox runtime layout contains a non-string path")
        path = PurePosixPath(value)
        is_root = path == workspace_root
        is_descendant = workspace_root in path.parents
        is_unsafe = path.is_absolute() or ".." in path.parts
        if is_unsafe or (is_root and not allow_root) or (not is_root and not is_descendant):
            raise SandboxCommandError(f"sandbox runtime layout contains unsafe path: {value!r}")
        return path.as_posix()

    async def _analysis_archive_spec(self) -> tuple[str, list[str]]:
        """Read and validate the worker-resolved workspace export rules."""
        assert self.backend is not None
        try:
            payload = json.loads((await self.backend.read_file(RUNTIME_LAYOUT)).decode("utf-8"))
        except FileNotFoundError as exc:
            raise SandboxCommandError("analysis export requires at least one completed job") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxCommandError(f"invalid sandbox runtime layout: {exc}") from exc

        if not isinstance(payload, dict):
            raise SandboxCommandError("sandbox runtime layout must be an object")
        expected_root = PurePosixPath(self.runtime_workspace).relative_to(CASE_ROOT)
        workspace_root_value = self._validated_workspace_path(
            payload.get("workspace_root"),
            expected_root,
            allow_root=True,
        )
        if PurePosixPath(workspace_root_value) != expected_root:
            raise SandboxCommandError(
                f"sandbox runtime layout changed workspace root: {workspace_root_value!r}",
            )

        values = payload.get("analysis_excludes")
        if not isinstance(values, list):
            raise SandboxCommandError("sandbox runtime layout has invalid analysis_excludes")
        excludes: list[str] = []
        for value in values:
            normalized = self._validated_workspace_path(value, expected_root, allow_root=False)
            if normalized not in excludes:
                excludes.append(normalized)
        return workspace_root_value, excludes

    async def _export(
        self,
        destination: str | Path,
        *,
        profile: Literal["analysis", "full"],
        include_candidate: bool,
    ) -> Path:
        """Export the active case while the caller holds the operation lock."""
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
            "container_id": self.container_id,
            "candidate_id": candidate_id,
            "candidate_mode": "source" if isinstance(self.candidate, SourceCandidate) else "image",
            "config": self.config,
            "environment_names": self.environment_names,
            "export_profile": profile,
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
        if profile == "analysis":
            workspace_root, analysis_excludes = await self._analysis_archive_spec()
            command = [
                "tar",
                "-czf",
                EXPORT_ARCHIVE,
                "-C",
                CASE_ROOT,
                *(f"--exclude={path}" for path in analysis_excludes),
                "manifest.json",
                "runtime-layout.json",
                "logs",
                "results",
                workspace_root,
            ]
        else:
            command = ["tar", "-czf", EXPORT_ARCHIVE, "-C", CASE_ROOT, "."]
        if include_candidate:
            command.extend(["-C", "/workspace", "candidate"])
        await self._exec_checked("export_case", command)
        payload = await self.backend.read_file(EXPORT_ARCHIVE)
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target

    async def reset_case(self, case_id: str) -> None:
        """Discard the active case and prepare this container for another.

        Candidate code, its virtual environment, and the uploaded worker stay
        intact. The runtime workspace, requests, logs, results, manifest, and
        previous export archive are removed. Operations are serialized so a
        reset cannot overlap a job or export.
        """
        _validate_case_id(case_id)
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            await self._exec_checked(
                "clear_case",
                ["rm", "-rf", CASE_ROOT, EXPORT_ARCHIVE, WORKSPACE_ARCHIVE, WORKSPACE_EXPORT_ARCHIVE],
            )
            await self._create_case_layout()
            self.case_id = case_id
            self.actions = []

    async def close(self) -> None:
        """Destroy the case container. Export before calling this method."""
        async with self._operation_lock:
            if not self._closed:
                self._closed = True
                await self.workspace.close()

    async def __aenter__(self) -> "DockerReMeSandbox":
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()
