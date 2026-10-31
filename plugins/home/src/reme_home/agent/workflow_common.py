"""Shared sandbox, tools and invocation helpers for Home workflow agents."""

import json
import sys
from pathlib import Path
from uuid import uuid4

from agentscope.tool import BackendBase, DirEntry, Glob, Grep, Read, Toolkit, Write
import yaml

from ..auto_bash import AutoBash
from ..code.repository import validate_repository
from ..evaluation.artifacts import contained, write_json
from ..evaluation.component import blocking
from .case_study import python_tool
from .sandbox import create_case_study_sandbox

INSPECTION_JOBS = (
    "home_list_validation_results",
    "home_inspect_validation",
    "home_trace_query_evidence",
    "home_inspect_memory_history",
    "home_compare_memory",
    "home_code_status",
    "home_list_candidate_branches",
    "home_code_diff",
    "home_list_code_tree",
    "home_read_code_file",
    "home_search_code",
)


def required_job(wrapper, name):
    """Do not silently bypass a disabled or unstarted Home Job."""
    job = wrapper.app_context.jobs[name]
    if not job.enabled or not job.is_started:
        raise RuntimeError(f"Workflow requires a started Job: {name}")
    return job


def create_toolkit(wrapper, backend, scope, *, extra_tools=(), job_names=()):
    """All builtin tools share the OS sandbox; privileged mutations use managed Jobs."""
    tools = [
        Read(backend=backend),
        Glob(backend=backend),
        Grep(backend=backend),
        Write(backend=backend),
        AutoBash(cwd=scope["scratch"], backend=backend, is_read_only=False),
        python_tool(backend, scope["scratch"]),
    ]
    tools[-1].description = (
        tools[-1]
        .description.replace(
            "No host path is writable.",
            "Only the host-granted scratch and candidate paths are writable.",
        )
        .replace("case directory", "scratch directory")
    )
    tools[-2].description = (
        "Run shell commands inside the OS file sandbox; network is denied. Use absolute allowed paths. "
        "Only private scratch and, for the optimizer, declared candidate paths are writable. "
        "Git metadata is read-only: commit, switch, create and merge only through Home Jobs. "
        "Run ReMe evaluations only through home_evaluate, never through shell or Python. "
        "Keep stderr visible and use set -o pipefail for pipelines. timeout is in milliseconds."
    )
    tools.extend(extra_tools)
    tools.extend(
        wrapper._make_tool(required_job(wrapper, name))  # pylint: disable=protected-access
        for name in (*INSPECTION_JOBS, *job_names)
    )
    for tool in tools:
        tool.is_concurrency_safe = False
    return Toolkit(tools=tools)


async def workflow_reply(
    wrapper,
    home,
    role,
    task,
    *,
    tool_factory=None,
    job_names=(),
    agent_options=None,
    memory_limit_mib=None,
):
    """Use fresh state, the dedicated configured model and sandboxed tools for every call."""
    scope = await blocking(prepare_scope, home, role)
    backend = WorkflowSandbox(scope, memory_limit_mib)
    probe = await backend.exec_shell(
        [sys.executable, "-I", "-B", "-c", "print('home workflow sandbox')"],
        timeout=10,
    )
    if not probe.ok():
        raise RuntimeError("Home workflow sandbox unavailable: " + probe.stderr.decode(errors="replace"))
    prompt = yaml.safe_load(Path(__file__).with_name(f"{role}.yaml").read_text(encoding="utf-8"))
    return await wrapper.reply(
        task,
        system_prompt=prompt["system_prompt"].format(**scope),
        toolkit=create_toolkit(
            wrapper,
            backend,
            scope,
            extra_tools=tool_factory(backend) if tool_factory is not None else (),
            job_names=job_names,
        ),
        builtin_tools=[],
        job_tools=[],
        skills=[],
        resume=None,
        session_id=None,
        fork_session=False,
        output_schema=None,
        permission_mode="bypass",
        **(agent_options or {}),
    )


def prepare_scope(home, role):
    """Grant evidence reads and private scratch writes, plus declared optimizer roots."""
    repo, record = validate_repository(home)
    scratch = contained(home, f"tmp/{role}-{uuid4().hex}", directory=True)
    scratch.mkdir()
    roots = [
        contained(home, name, directory=True)
        for name in ("code", "datasets", "docs", "evaluations", "logs", "sampling")
    ]
    roots.append(contained(home, "metadata/preparation.json"))
    writable = [scratch]
    if role == "optimizer":
        for value in record["code"]["candidate_paths"]:
            path = contained(repo, value)
            if ".git" in path.relative_to(repo).parts or path == repo:
                raise ValueError("Optimizer cannot write Git metadata or the repository root")
            writable.append(path)
    return {
        "home": str(home),
        "code_repository": str(repo),
        "scratch": str(scratch),
        "docs": str(home / "docs"),
        "read_only_paths": [str(path) for path in roots],
        "writable_paths": [str(path) for path in writable],
    }


def save_record(home, category, payload):
    """Write a uniquely named host-owned provenance record, outside agent write grants."""
    path = contained(home, f"logs/{category}/{uuid4().hex}.json")
    write_json(path, payload)
    return str(path)


_INSPECT = """\
import json, pathlib, sys
path = pathlib.Path(sys.argv[2])
def entry(p):
    s = p.stat()
    return dict(name=p.name, is_dir=p.is_dir(), size_bytes=None if p.is_dir() else s.st_size, mtime=s.st_mtime)
if sys.argv[1] == 'stat':
    print(json.dumps(entry(path)))
else:
    result = []
    for p in path.iterdir():
        try:
            result.append(entry(p))
        except FileNotFoundError:
            pass
    print(json.dumps(result))
"""


class WorkflowSandbox(BackendBase):
    """Route every builtin I/O primitive through Home's OS sandbox, including metadata.

    Rebuild the backend before each operation: managed branch switches can
    replace an allowed file's inode. Core validation still checks paths and
    identities for each subprocess; no operation falls back to local I/O.
    """

    def __init__(self, scope, memory_limit_mib=None):
        self.scope = scope
        self.memory_limit_mib = memory_limit_mib

    def _backend(self):
        return create_case_study_sandbox(
            cwd=self.scope["scratch"],
            read_only_paths=self.scope["read_only_paths"],
            writable_paths=self.scope["writable_paths"],
            memory_limit_mib=self.memory_limit_mib,
        )

    async def getcwd(self):
        """Use private scratch as the default, never the host process directory."""
        return self.scope["scratch"]

    async def exec_shell(self, command, *, cwd=None, timeout=None):
        """Keep command execution, environment and process cleanup inside isolation."""
        return await self._backend().exec_shell(command, cwd=cwd, timeout=timeout)

    async def read_file(self, path):
        """Use core sandboxed file reads, not LocalBackend's host reads."""
        return await self._backend().backend.read_file(path)

    async def write_file(self, path, data):
        """Use core sandboxed writes, including parent creation and stdin transport."""
        await self._backend().backend.write_file(path, data)

    async def _inspect(self, operation, path):
        result = await self.exec_shell([sys.executable, "-I", "-B", "-c", _INSPECT, operation, path], timeout=10)
        if not result.ok():
            raise OSError(result.stderr.decode(errors="replace"))
        return json.loads(result.stdout)

    async def stat(self, path):
        """Use portable sandboxed metadata on Linux and macOS."""
        return DirEntry(**await self._inspect("stat", path))

    async def scandir(self, path):
        """Avoid GNU find assumptions and host filesystem metadata access."""
        return [DirEntry(**item) for item in await self._inspect("scandir", path)]

    async def stat_mtime(self, path):
        """Preserve the builtin Edit/Write optimistic modification checks."""
        return (await self.stat(path)).mtime

    async def list_dir(self, path, *, recursive=False):
        """Use the portable metadata implementation for immediate children."""
        if recursive:
            return await super().list_dir(path, recursive=True)
        return [entry.name for entry in await self.scandir(path)]
