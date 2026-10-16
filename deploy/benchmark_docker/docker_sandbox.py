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

from ._cleanup import cleanup_on_error
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
_CASE_ARCHIVE_FORMAT = "reme-case-v1"
_CLEANUP_TIMEOUT = 30.0
_LOG_TIMEOUT = 5.0
_PRIVATE_EVALUATION_FIELDS = frozenset(
    {"query", "question", "golden_answer", "question_type", "rubric", "judge_arguments"},
)


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

    async def stop(self) -> None:
        """Force-stop container processes while keeping files available."""
        raise NotImplementedError

    def get_backend(self) -> Backend:
        """Return the execution backend."""
        raise NotImplementedError


WorkspaceBuilder = Callable[[str, str, dict[str, str]], Workspace]


class SandboxCommandError(RuntimeError):
    """A required command failed inside a case sandbox."""


class SandboxCommandTimeoutError(SandboxCommandError):
    """A command timed out; cleanup failures are attached as exception notes."""


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


def _case_archive_workspace_root(
    archive: tarfile.TarFile,
    members: dict[PurePosixPath, tarfile.TarInfo],
) -> PurePosixPath | None:
    """Distinguish marked case artifacts from raw workspace contents."""
    workspace_root = PurePosixPath("reme_workspace")
    has_workspace = any(path.parts[0] == workspace_root.name for path in members)
    manifest = members.get(PurePosixPath("manifest.json"))
    if manifest is None:
        if has_workspace:
            raise ValueError(
                "unmarked case archive; upload its reme_workspace directory or recreate it with a current case export",
            )
        return None
    if not manifest.isfile():
        raise ValueError("workspace archive manifest.json must be a regular file")
    content = archive.extractfile(manifest)
    assert content is not None
    try:
        payload = json.loads(content.read())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if has_workspace:
            raise ValueError("case archive contains an invalid manifest.json") from exc
        return None  # A raw workspace may contain an unrelated manifest file.
    if not isinstance(payload, dict) or "archive_format" not in payload:
        if has_workspace and not isinstance(payload, dict):
            raise ValueError("case archive manifest.json must be an object")
        # Older full case exports have a manifest without archive_format.
        return workspace_root if has_workspace else None
    if payload["archive_format"] != _CASE_ARCHIVE_FORMAT:
        raise ValueError(f"unsupported case archive format: {payload['archive_format']!r}")
    root = payload.get("workspace_root")
    if root is None:
        raise ValueError("case archive has no reusable workspace")
    if root != workspace_root.as_posix():
        raise ValueError(f"case archive has an invalid workspace root: {root!r}")
    if not has_workspace:
        raise ValueError("case archive workspace is missing")
    return workspace_root


def _archive_workspace(source: str | Path) -> bytes:
    """Create a safe, deterministic gzip tar archive from a host workspace.

    ``source`` may be a workspace directory or an archive previously returned
    by :meth:`DockerReMeSandbox.export_workspace`, or a marked case archive
    containing a workspace. Incoming archives are read and rebuilt so links,
    special files, and traversal paths never reach the sandbox extractor.
    """
    supplied_path = Path(source).expanduser()
    if supplied_path.is_symlink():
        raise ValueError("workspace source must not be a symbolic link")
    source_path = supplied_path.resolve(strict=True)
    entries: list[tuple[tarfile.TarInfo, bytes | None]] = []
    if source_path.is_dir():
        for item in sorted(source_path.rglob("*"), key=lambda value: value.relative_to(source_path).as_posix()):
            relative = item.relative_to(source_path)
            if item.is_symlink():
                raise ValueError(f"workspace source contains a symbolic link: {relative}")
            path = _workspace_member_path(relative.as_posix())
            assert path is not None
            info = tarfile.TarInfo(path.as_posix())
            info.mode = item.stat().st_mode & 0o777
            if item.is_dir():
                info.type = tarfile.DIRTYPE
                data = None
            elif item.is_file():
                data = item.read_bytes()
                info.size = len(data)
            else:
                raise ValueError(f"workspace source contains unsupported file: {relative}")
            entries.append((info, data))
    elif source_path.is_file():
        with tarfile.open(source_path, mode="r:*") as archive:
            members: dict[PurePosixPath, tarfile.TarInfo] = {}
            for member in archive.getmembers():
                path = _workspace_member_path(member.name)
                if path is None:
                    continue
                if path in members:
                    raise ValueError(f"workspace archive contains duplicate path: {path}")
                members[path] = member
            workspace_root = _case_archive_workspace_root(archive, members)
            for path, member in members.items():
                if workspace_root is not None:
                    if path != workspace_root and workspace_root not in path.parents:
                        continue
                    if path == workspace_root:
                        if not member.isdir():
                            raise ValueError("case export has a non-directory reme_workspace entry")
                        continue
                    path = path.relative_to(workspace_root)
                info = tarfile.TarInfo(path.as_posix())
                info.mode = member.mode & 0o777
                if member.isdir():
                    info.type = tarfile.DIRTYPE
                    data = None
                elif member.isfile():
                    content = archive.extractfile(member)
                    if content is None:
                        raise ValueError(f"workspace archive cannot read file: {path}")
                    data = content.read()
                    info.size = len(data)
                else:
                    raise ValueError(f"workspace archive contains unsupported entry: {member.name!r}")
                entries.append((info, data))
    else:
        raise ValueError(f"workspace source is neither a directory nor a file: {source_path}")

    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for info, data in entries:
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data) if data is not None else None)
    return raw.getvalue()


def _default_workspace_builder(workspace_id: str, image: str, env: dict[str, str]) -> Workspace:
    """Create the AgentScope 2.0.9 Docker backend lazily."""
    import agentscope

    from .direct_docker_workspace import DirectDockerWorkspace

    if agentscope.__version__ != "2.0.9":
        raise RuntimeError(f"benchmark Docker host requires agentscope 2.0.9, got {agentscope.__version__}")
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
        config: str = "benchmark.yaml",
        plugins: tuple[str, ...] | list[str] = (),
        command_timeout: float = 1800.0,
        workspace_builder: WorkspaceBuilder | None = None,
    ) -> None:
        self.candidate = candidate
        self.env = dict(env or {})
        self.config = config
        self.plugins = tuple(plugins)
        if not all(isinstance(name, str) and name for name in self.plugins):
            raise TypeError("sandbox plugins must be non-empty strings")
        if len(self.plugins) != len(set(self.plugins)):
            raise ValueError("sandbox plugins must be unique")
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
            plugins=self.plugins,
            command_timeout=self.command_timeout,
            environment_names=sorted(self.env),
        )
        try:
            await case.initialize()
        except BaseException as exc:
            await cleanup_on_error([(f"workspace {container_id!r}", workspace.close())], exc)
            raise
        return case

    async def create_cases(self, case_ids: list[str], *, concurrency: int = 4) -> list["DockerReMeSandbox"]:
        """Create several independent cases while bounding Docker pressure."""
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_ids must be unique")
        semaphore = asyncio.Semaphore(concurrency)

        async def create(case_id: str) -> DockerReMeSandbox | BaseException:
            try:
                async with semaphore:
                    return await self.create_case(case_id)
            except BaseException as exc:
                # Preserve cleanup notes on CancelledError instead of letting
                # gather replace it with a fresh cancellation exception.
                return exc

        tasks = [asyncio.create_task(create(case_id)) for case_id in case_ids]
        error: BaseException | None = None
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException as exc:
            error = exc
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
        created = [result for result in results if isinstance(result, DockerReMeSandbox)]
        failures = [result for result in results if isinstance(result, BaseException)]
        if error is None:
            if not failures:
                return created
            error = failures.pop(0)
            for failure in failures:
                error.add_note(f"additional creation failure: {type(failure).__name__}: {failure}")
        for failure in failures:
            for note in getattr(failure, "__notes__", ()):
                error.add_note(note)
        await cleanup_on_error(((f"workspace {case.container_id!r}", case.close()) for case in created), error)
        raise error


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
        plugins: tuple[str, ...],
        command_timeout: float,
        environment_names: list[str],
    ) -> None:
        self.case_id = case_id
        self.container_id = container_id
        self.candidate = candidate
        self.workspace = workspace
        self.config = config
        self.plugins = plugins
        self.command_timeout = command_timeout
        self.environment_names = environment_names
        self.backend: Backend | None = None
        self.python = "python3"
        self.actions: list[ActionRecord] = []
        self._closed = False
        self._reusable = True
        self._operation_lock = asyncio.Lock()
        self._build_log: bytes | None = None

    @property
    def reusable(self) -> bool:
        """Whether another case can safely use this container."""
        return self._reusable and not self._closed

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
        await self._write_file(f"{HARNESS_ROOT}/worker.py", worker)

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
        await self._write_file(SOURCE_ARCHIVE, self.candidate.snapshot.archive)
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
        for index, project in enumerate(self.candidate.install_projects):
            project_path = CANDIDATE_ROOT if project == "." else f"{CANDIDATE_ROOT}/{project}"
            await self._exec_checked(
                "install_candidate" if index == 0 else f"install_candidate:{project}",
                [
                    self.python,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "--editable",
                    project_path,
                ],
            )
        await self._validate_candidate(self.python)

    async def _validate_candidate(self, python: str) -> None:
        plugin_names = json.dumps(self.plugins)
        script = (
            "import agentscope, importlib, json, reme; "
            "from importlib.metadata import entry_points; "
            "from reme.plugin import PluginManager; "
            f"names={plugin_names}; "
            "eps=entry_points().select(group='reme.plugins'); "
            "selected={name: next((ep for ep in eps if ep.name == name), None) for name in names}; "
            "PluginManager.discover(names); "
            "paths={name: importlib.import_module(ep.value.split(':', 1)[0]).__file__ "
            "for name, ep in selected.items() if ep is not None}; "
            "print(json.dumps({'reme': reme.__file__, 'agentscope': agentscope.__version__, 'plugins': paths}))"
        )
        result = await self._exec_checked("validate_candidate", [python, "-c", script])
        validation = json.loads(_decode(result.stdout))
        if validation.get("agentscope") != "2.0.9":
            raise SandboxCommandError(
                f"candidate environment requires agentscope 2.0.9, got {validation.get('agentscope')!r}",
            )
        candidate_root = PurePosixPath(CANDIDATE_ROOT)

        def is_candidate_path(value: Any) -> bool:
            return isinstance(value, str) and candidate_root in PurePosixPath(value).parents

        if isinstance(self.candidate, SourceCandidate) and not is_candidate_path(validation.get("reme")):
            raise SandboxCommandError(
                f"candidate import did not resolve under {CANDIDATE_ROOT}: {validation.get('reme')}",
            )
        plugin_paths = validation.get("plugins")
        if not isinstance(plugin_paths, dict) or set(plugin_paths) != set(self.plugins):
            expected = list(self.plugins)
            raise SandboxCommandError(
                f"candidate did not expose the requested plugins: expected {expected!r}, got {plugin_paths!r}",
            )
        if isinstance(self.candidate, SourceCandidate):
            escaped = {name: path for name, path in plugin_paths.items() if not is_candidate_path(path)}
            if escaped:
                raise SandboxCommandError(f"candidate plugin imports did not resolve under {CANDIDATE_ROOT}: {escaped}")

    async def _write_file(self, path: str, data: bytes) -> None:
        """Reject writes after an interrupted or uncertain Docker upload."""
        assert self.backend is not None, "sandbox is not initialized"
        if not self.reusable:
            raise SandboxCommandError("sandbox container is no longer reusable")
        try:
            await self.backend.write_file(path, data)
        except (Exception, asyncio.CancelledError):
            # Docker may have accepted the upload before the client failed.
            self._reusable = False
            raise

    async def _exec(self, name: str, command: list[str], *, cwd: str | None = None) -> Any:
        assert self.backend is not None, "sandbox is not initialized"
        if not self.reusable:
            raise SandboxCommandError("sandbox container is no longer reusable")
        started = _utc_now()
        try:
            async with asyncio.timeout(self.command_timeout):
                result = await self.backend.exec_shell(command, cwd=cwd, timeout=self.command_timeout)
        except TimeoutError as exc:
            error = SandboxCommandTimeoutError(f"sandbox action {name!r} timed out")
            await cleanup_on_error([(f"container {self.container_id!r}", self._abort())], error)
            raise error from exc
        except (Exception, asyncio.CancelledError) as exc:
            # A backend failure may occur after Docker started the command.
            await cleanup_on_error([(f"container {self.container_id!r}", self._abort())], exc)
            raise
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
        if result.exit_code == -1:
            error = SandboxCommandTimeoutError(f"sandbox action {name!r} timed out")
            await cleanup_on_error([(f"container {self.container_id!r}", self._abort())], error)
            raise error
        return result

    async def _abort(self) -> None:
        """Retire an uncertain command while its caller holds the operation lock."""
        self._reusable = False
        error = None
        try:
            await asyncio.wait_for(self.workspace.stop(), timeout=_CLEANUP_TIMEOUT)
            # Read files only after the candidate has stopped. Diagnostics must
            # never delay termination or prevent container removal.
            try:
                assert self.backend is not None
                self._build_log = await asyncio.wait_for(
                    self.backend.read_file(f"{CASE_ROOT}/build_log/build.log"),
                    timeout=_LOG_TIMEOUT,
                )
            except Exception:
                pass
        except BaseException as exc:
            error = exc
            raise
        finally:
            if error is None:
                await self._close_workspace()
            else:
                await cleanup_on_error([("container removal", self._close_workspace())], error)

    async def _exec_checked(self, name: str, command: list[str], *, cwd: str | None = None) -> Any:
        result = await self._exec(name, command, cwd=cwd)
        if not result.ok():
            raise SandboxCommandError(
                f"sandbox action {name!r} failed with exit {result.exit_code}: {_decode(result.stderr)}",
            )
        return result

    async def _run_job(self, job: str, arguments: dict[str, Any] | None = None) -> JobResult:
        """Run one ReMe job while the caller holds the operation lock."""
        response = await self._run_worker_request("job", {"job": job, "arguments": arguments or {}})
        return JobResult.from_dict(response)

    async def _run_worker_request(self, mode: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one structured worker request and return its JSON response."""
        assert self.backend is not None, "sandbox is not initialized"
        request_id = uuid.uuid4().hex
        request_path = f"{CASE_ROOT}/inbox/{request_id}.json"
        response_path = f"{self.results_dir}/{request_id}.json"
        request = {
            "mode": mode,
            "config": self.config,
            "plugins": list(self.plugins),
            "case_id": self.case_id,
            "case_root": CASE_ROOT,
            "workspace_dir": self.runtime_workspace,
            **payload,
        }
        await self._write_file(request_path, json.dumps(request, ensure_ascii=False).encode("utf-8"))
        action_name = f"job:{payload['job']}" if mode == "job" else f"worker:{mode}"
        await self._exec(
            action_name,
            [self.python, f"{HARNESS_ROOT}/worker.py", "--request", request_path, "--response", response_path],
            cwd=CASE_ROOT,
        )
        try:
            result = json.loads((await self.backend.read_file(response_path)).decode("utf-8"))
        except FileNotFoundError as exc:
            description = f"job {payload['job']!r}" if mode == "job" else f"worker mode {mode!r}"
            raise SandboxCommandError(f"{description} did not produce {response_path}") from exc
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
            leaked = _PRIVATE_EVALUATION_FIELDS.intersection(value.get("arguments", {}))
            if leaked:
                names = ", ".join(sorted(leaked))
                raise ValueError(f"construction job arguments contain private query fields: {names}")
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
        """Answer multiple queries in one candidate Application with isolated logs."""
        serialized, query_ids = self._serialize_queries(queries)
        if not serialized:
            raise ValueError("run_queries requires at least one query")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query IDs must be unique")
        async with self._operation_lock:
            return await self._run_worker_request("queries", {"queries": serialized})

    async def run_query(self, query: EvaluationQuery | dict[str, Any]) -> dict[str, Any]:
        """Answer one append-only query for a host-side lease."""

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
            value = (
                {
                    "query_id": query.query_id,
                    "question": query.question,
                    "answer_arguments": query.answer_arguments,
                    "answer_job": query.answer_job,
                }
                if isinstance(query, EvaluationQuery)
                else dict(query)
            )
            query_id = value.get("query_id")
            if not isinstance(query_id, str):
                raise ValueError("every evaluation query requires a string query_id")
            self._validate_query_artifact_id(query_id)
            if not isinstance(value.get("question"), str):
                raise ValueError(f"query {query_id!r} requires a string question")
            answer_arguments = value.get("answer_arguments") or {}
            if not isinstance(answer_arguments, dict):
                raise ValueError(f"query {query_id!r} answer_arguments must be an object")
            leaked = _PRIVATE_EVALUATION_FIELDS.intersection(answer_arguments)
            if leaked:
                names = ", ".join(sorted(leaked))
                raise ValueError(f"query {query_id!r} answer_arguments contain private judge fields: {names}")
            query_ids.append(query_id)
            # This allowlist is a security boundary.  Golden answers, rubrics,
            # question types and scoring configuration remain on the host and
            # are not written to the candidate container's inbox.
            serialized.append(
                {
                    "query_id": query_id,
                    "question": value["question"],
                    "answer_arguments": dict(answer_arguments),
                    "answer_job": str(value.get("answer_job") or "agentic_answer"),
                },
            )
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
        """Create a host-requested checkpoint containing durable file-native memory."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("memory history commit message must be a non-empty string")
        async with self._operation_lock:
            result = await self._run_worker_request("checkpoint", {"message": message})
            if result.get("success") is not True:
                raise SandboxCommandError(result.get("error") or "memory history checkpoint failed")

    async def answer(self, *, query: str, query_time: str = "") -> JobResult:
        """Generate an answer with the candidate's agentic_answer job."""
        async with self._operation_lock:
            result = await self._run_job("agentic_answer", {"query": query, "query_time": query_time})
            await self._write_file(
                f"{self.results_dir}/answer.json",
                json.dumps(asdict(result), ensure_ascii=False, default=str).encode("utf-8"),
            )
            return result

    async def export(
        self,
        destination: str | Path,
        *,
        profile: Literal["analysis", "full"] = "analysis",
        include_candidate: bool = False,
    ) -> Path:
        """Download the case archive and, for analysis, separate build/query logs.

        Analysis logs are saved under ``<destination>.artifacts`` without
        changing the archive's contents.
        """
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
            return await self._export_archive(
                destination,
                "export_workspace",
                self.runtime_workspace,
                ["."],
                archive_path=WORKSPACE_EXPORT_ARCHIVE,
            )

    async def export_build_log(self, destination: str | Path) -> Path:
        """Download the memory-construction log without wrapping it in an archive."""
        async with self._operation_lock:
            content = await self._read_build_log()
            return self._save_host_file(destination, content)

    async def read_build_log(self) -> str:
        """Return the build log, including when construction failed."""
        async with self._operation_lock:
            return (await self._read_build_log()).decode("utf-8", errors="replace")

    async def _read_build_log(self) -> bytes:
        if self._build_log is not None:
            return self._build_log
        if not self.reusable:
            raise SandboxCommandError("build log unavailable after interrupted command")
        return await self._require_artifact(f"{CASE_ROOT}/build_log/build.log", "build log")

    async def _require_artifact(self, path: str, description: str) -> bytes:
        """Read a required sandbox artifact, preserving its export-specific error."""
        assert self.backend is not None, "sandbox is not initialized"
        try:
            return await self.backend.read_file(path)
        except FileNotFoundError as exc:
            raise SandboxCommandError(f"{description} is missing required artifact: {path}") from exc

    async def export_memory_construction(self, destination: str | Path) -> Path:
        """Download the workspace and build log; reusable with upload_workspace."""
        async with self._operation_lock:
            return await self._export_archive(
                destination,
                "export_memory_construction",
                CASE_ROOT,
                ["reme_workspace", "build_log"],
                required=("build_log/build.log", "memory construction export"),
            )

    async def export_queries(self, destination: str | Path) -> Path:
        """Download per-query logs, results, and their aggregate summary."""
        async with self._operation_lock:
            return await self._export_archive(
                destination,
                "export_queries",
                CASE_ROOT,
                ["queries"],
                required=("queries/summary.json", "query export"),
            )

    async def export_query(self, query_id: str, destination: str | Path) -> Path:
        """Download one leased query's logs and result without a case summary."""

        self._validate_query_artifact_id(query_id)
        async with self._operation_lock:
            return await self._export_archive(
                destination,
                "export_query",
                CASE_ROOT,
                [f"queries/{query_id}"],
                required=(f"queries/{query_id}/result.json", "query export"),
            )

    async def export_evaluation(self, destination: str | Path) -> Path:
        """Download reusable workspace and query artifacts, plus any build log."""
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            await self._require_artifact(f"{CASE_ROOT}/queries/summary.json", "evaluation export")
            members = ["reme_workspace", "queries"]
            try:
                await self.backend.read_file(f"{CASE_ROOT}/build_log/build.log")
            except FileNotFoundError:
                pass
            else:
                members.append("build_log")
            return await self._export_archive(destination, "export_evaluation", CASE_ROOT, members)

    @staticmethod
    def _save_host_file(destination: str | Path, content: bytes) -> Path:
        target = Path(destination).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    async def _export_archive(
        self,
        destination: str | Path,
        name: str,
        root: str,
        members: list[str],
        *,
        archive_path: str = EXPORT_ARCHIVE,
        required: tuple[str, str] | None = None,
    ) -> Path:
        """Archive selected sandbox paths and download the archive while locked."""
        assert self.backend is not None, "sandbox is not initialized"
        if required is not None:
            path, description = required
            await self._require_artifact(f"{root}/{path}", description)
        if root == CASE_ROOT:
            await self._write_export_manifest(
                name.removeprefix("export_"),
                workspace_root="reme_workspace" if "reme_workspace" in members else None,
            )
            members = ["manifest.json", *members]
        await self._exec_checked(name, ["tar", "-czf", archive_path, "-C", root, *members])
        return self._save_host_file(destination, await self.backend.read_file(archive_path))

    async def upload_workspace(self, source: str | Path, *, clear: bool = True) -> None:
        """Copy a host workspace directory or exported memory archive into this case.

        Case archives restore only their workspace; logs and query artifacts
        are excluded. Query-only archives cannot seed a runtime workspace.
        By default the existing runtime workspace is deleted first, ensuring a
        query case receives only the supplied memory state. Set ``clear=False``
        only when intentionally merging files into the current workspace.
        The copy is made through a validated archive, so each sandbox receives
        an independent filesystem tree and can run queries in parallel.
        """
        archive = await asyncio.to_thread(_archive_workspace, source)
        async with self._operation_lock:
            assert self.backend is not None, "sandbox is not initialized"
            if clear:
                await self._exec_checked("clear_runtime_workspace", ["rm", "-rf", self.runtime_workspace])
            await self._exec_checked("create_runtime_workspace", ["mkdir", "-p", self.runtime_workspace])
            await self._write_file(WORKSPACE_ARCHIVE, archive)
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

    async def _write_export_manifest(self, profile: str, *, workspace_root: str | None) -> None:
        """Identify case archive layout without adding artifacts to its workspace."""
        candidate_id = (
            self.candidate.candidate_id
            if isinstance(self.candidate, SourceCandidate)
            else self.candidate.resolved_candidate_id
        )
        manifest = {
            "archive_format": _CASE_ARCHIVE_FORMAT,
            "workspace_root": workspace_root,
            "case_id": self.case_id,
            "container_id": self.container_id,
            "candidate_id": candidate_id,
            "candidate_mode": "source" if isinstance(self.candidate, SourceCandidate) else "image",
            "config": self.config,
            "plugins": list(self.plugins),
            "install_projects": (
                list(self.candidate.install_projects) if isinstance(self.candidate, SourceCandidate) else []
            ),
            "environment_names": self.environment_names,
            "export_profile": profile,
            "exported_at": _utc_now(),
        }
        await self._write_file(
            f"{CASE_ROOT}/manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    async def _export(
        self,
        destination: str | Path,
        *,
        profile: Literal["analysis", "full"],
        include_candidate: bool,
    ) -> Path:
        """Export the active case while the caller holds the operation lock."""
        assert self.backend is not None, "sandbox is not initialized"
        target = Path(destination).expanduser().resolve()
        artifacts = Path(f"{target}.artifacts")
        if profile == "analysis" and artifacts.exists():
            raise FileExistsError(f"analysis artifacts already exist: {artifacts}")
        if include_candidate and isinstance(self.candidate, ImageCandidate):
            raise ValueError("an image candidate has no source directory to export")
        await self._write_export_manifest(profile, workspace_root="reme_workspace")
        await self._write_file(
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
        self._save_host_file(target, await self.backend.read_file(EXPORT_ARCHIVE))
        if profile == "analysis":
            await self._export_analysis_logs(artifacts)
        return target

    async def _export_analysis_logs(self, destination: Path) -> None:
        """Save available build and query artifacts alongside the analysis archive."""
        assert self.backend is not None
        try:
            build_log = await self.backend.read_file(f"{CASE_ROOT}/build_log/build.log")
        except FileNotFoundError:
            build_log = None
        if build_log is not None:
            log_path = destination / "build_log" / "build.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_bytes(build_log)

        try:
            await self.backend.read_file(f"{CASE_ROOT}/queries/summary.json")
        except FileNotFoundError:
            return
        await self._exec_checked(
            "export_analysis_queries",
            ["tar", "-czf", EXPORT_ARCHIVE, "-C", CASE_ROOT, "queries"],
        )
        archive = await self.backend.read_file(EXPORT_ARCHIVE)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
            for member in source:
                path = _workspace_member_path(member.name)
                if path is None or not path.parts or path.parts[0] != "queries":
                    raise SandboxCommandError(f"query export contains unsafe path: {member.name!r}")
                output = destination.joinpath(*path.parts)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    content = source.extractfile(member)
                    if content is None:
                        raise SandboxCommandError(f"query export cannot read: {member.name!r}")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(content.read())
                else:
                    raise SandboxCommandError(f"query export contains unsupported member: {member.name!r}")

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
            await self._close_workspace()

    async def _close_workspace(self) -> None:
        """Close without acquiring the operation lock a second time."""
        if not self._closed:
            self._reusable = False
            await asyncio.wait_for(self.workspace.close(), timeout=_CLEANUP_TIMEOUT)
            self._closed = True

    async def __aenter__(self) -> "DockerReMeSandbox":
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()
