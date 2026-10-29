"""One case-local investigation using ReMe Jobs and sandboxed command tools."""

import sys
from pathlib import Path

from agentscope.tool import FunctionTool, Toolkit
from pydantic import Field, field_validator
import yaml

from reme.steps.base_step import BaseStep, Ref
from reme.utils.file_sandbox import create_file_sandbox_backend

from ..auto_bash import AutoBash
from ..code.repository import resolve_revision, validate_repository
from ..evaluation.artifacts import contained
from ..evaluation.component import blocking
from ..evaluation.evidence_step import _source_case
from ..evaluation.inputs import entity_id
from ..evaluation.memory_inspection_step import _repository
from ..preparation.spec import SpecModel
from ..workspace import HomeWorkspace
from .wrapper import BoundedAgentWrapper

PROMPT = Path(__file__).with_suffix(".yaml")
CASE_TOOLS = (
    "home_inspect_validation",
    "home_trace_query_evidence",
    "home_inspect_memory_history",
    "home_compare_memory",
)
CODE_TOOLS = ("home_list_code_tree", "home_search_code", "home_read_code_file")


class CaseStudyInput(SpecModel):
    """Select one existing validation case; runtime and permissions are host-owned."""

    validation_path: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    focus: str = Field(
        default="Investigate the failed or low-scoring queries and their earliest supported causes.",
        min_length=1,
    )

    @field_validator("case_id")
    @classmethod
    def validate_case(cls, value):
        """Keep the case inside its manifest-selected artifact directory."""
        return entity_id(value)


def prepare_scope(home, spec):
    """Resolve evidence and the validation's immutable candidate revision before model calls."""
    run, manifest, _memory = _repository(home, spec.validation_path, spec.case_id)
    case_path = contained(run, f"cases/{spec.case_id}", directory=True)
    if not case_path.is_dir():
        raise ValueError("Case artifacts do not exist")
    source_path, _case = _source_case(home, manifest, spec.case_id)
    repo, _record = validate_repository(home)
    commit = manifest.get("commit_sha")
    if not isinstance(commit, str) or len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("Evaluation manifest must record a full candidate commit_sha")
    revision = resolve_revision(repo, commit)
    roots = [
        case_path,
        source_path,
        contained(run, "manifest.json"),
        contained(home, "code", directory=True),
        contained(home, "docs", directory=True),
    ]
    return {
        "validation_path": str(run),
        "case_id": spec.case_id,
        "case_path": str(case_path),
        "candidate_commit": revision,
        "code_repository": str(repo),
        "source_case": str(source_path),
        "docs": str(home / "docs"),
        "read_only_paths": [str(path) for path in roots],
    }


def python_tool(backend, cwd):
    """Run source or an existing script with the same read-only sandbox as AutoBash."""

    async def python_execute(
        code: str | None = None,
        timeout: float = 60,
        *,
        file: str | None = None,
        args: list[str] | None = None,
    ) -> dict:
        """Execute code or file, exclusively; print results to stdout. No host path is writable.

        Args:
            code: Python source, mutually exclusive with file.
            timeout: Execution timeout in seconds, greater than zero and at most 600.
            file: Existing script path, absolute or relative to the case directory; sandbox read rules apply.
            args: String arguments for file mode only.
        """
        if (code is not None and not isinstance(code, str)) or (file is not None and not isinstance(file, str)):
            raise ValueError("code and file must be strings or null")
        has_code, has_file = bool(code and code.strip()), bool(file and file.strip())
        if has_code == has_file:
            raise ValueError("Provide exactly one of code or file")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
            raise ValueError("timeout must be greater than zero and at most 600 seconds")
        if args is not None and (
            not has_file or not isinstance(args, list) or any(not isinstance(a, str) for a in args)
        ):
            raise ValueError("args must be a list of strings in file mode only")
        command = [sys.executable, "-I", "-B"]
        command += ["-c", code] if has_code else [str(Path(cwd) / file), *(args or [])]
        result = await backend.exec_shell(command, cwd=cwd, timeout=timeout)
        return {
            "returncode": result.exit_code,
            "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace"),
            "timed_out": result.exit_code == -1 and result.stderr == b"timed out",
        }

    return FunctionTool(python_execute, is_concurrency_safe=False)


def create_toolkit(wrapper, scope, backend):
    """Expose only read-only Jobs with hidden, enforced scope and two sandboxed executors."""
    tools = []
    for name in CASE_TOOLS + CODE_TOOLS:
        job = wrapper.app_context.jobs[name]
        if not job.enabled or not job.is_started:
            raise RuntimeError(f"Case study requires a started Job: {name}")
        injected = (
            {"revision": scope["candidate_commit"]}
            if name in CODE_TOOLS
            else {"validation_path": scope["validation_path"], "case_id": scope["case_id"]}
        )
        if name == "home_compare_memory":
            injected["target_validation_path"] = None
        tools.append(wrapper._make_tool(job, injected_job_kwargs=injected))  # pylint: disable=protected-access
    tools.extend(
        [
            AutoBash(cwd=scope["case_path"], backend=backend, is_read_only=False),
            python_tool(backend, scope["case_path"]),
        ],
    )
    for tool in tools:
        tool.is_concurrency_safe = False
    return Toolkit(tools=tools)


class HomeCaseStudyStep(BaseStep):
    """Run one fresh agent through the application's bounded wrapper and model lifecycle."""

    workspace: HomeWorkspace = Ref(HomeWorkspace, "home.workspace")
    agent_wrapper: BoundedAgentWrapper = Ref(BoundedAgentWrapper, "agent_wrapper")

    async def execute(self):
        """Prepare scope, require isolation, then return the case report and framework token usage."""
        if self.context.get("_allowed_paths") is not None:
            raise ValueError("Home case study requires unrestricted access to its managed artifacts")
        if not self.workspace.is_started or not self.agent_wrapper.is_started:
            raise RuntimeError("Start the Home workspace and case-study agent wrapper first")
        spec = CaseStudyInput.model_validate(
            {key: self.context[key] for key in CaseStudyInput.model_fields if key in self.context},
        )
        scope = await blocking(prepare_scope, self.workspace.home_path, spec)
        backend = create_file_sandbox_backend(
            cwd=scope["case_path"],
            read_only_paths=scope["read_only_paths"],
            writable_paths=[],
            memory_limit_mib=self.kwargs.get("memory_limit_mib"),
        )
        # Fail before spending model tokens if the OS sandbox or its runtime cannot start.
        probe = await backend.exec_shell(
            [sys.executable, "-I", "-B", "-c", "pass"],
            cwd=scope["case_path"],
            timeout=10,
        )
        if not probe.ok():
            raise RuntimeError("Case-study sandbox unavailable: " + probe.stderr.decode(errors="replace"))
        toolkit = create_toolkit(self.agent_wrapper, scope, backend)
        output_limit = self.kwargs.get("output_token_limit", 2500)
        if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit < 1:
            raise ValueError("output_token_limit must be a positive integer")
        prompts = yaml.safe_load(PROMPT.read_text(encoding="utf-8"))
        if not isinstance(prompts, dict) or any(
            not isinstance(prompts.get(key), str) or not prompts[key].strip()
            for key in ("system_prompt", "task_prompt")
        ):
            raise ValueError("Case-study YAML requires nonempty system_prompt and task_prompt strings")
        prompt = prompts["system_prompt"].format(**scope, output_token_limit=output_limit)
        result = await self.agent_wrapper.reply(
            prompts["task_prompt"].format(focus=spec.focus),
            system_prompt=prompt,
            toolkit=toolkit,
            builtin_tools=[],
            job_tools=[],
            skills=[],
            resume=None,
            session_id=None,
            fork_session=False,
            output_schema=None,
            permission_mode="bypass",
        )
        self.context.response.answer = result["result"]
        self.context.response.metadata.update(
            {key: scope[key] for key in ("validation_path", "case_id", "candidate_commit")},
        )
        self.context.response.metadata.update({key: result[key] for key in ("session_id", "usage") if key in result})
        return self.context.response
