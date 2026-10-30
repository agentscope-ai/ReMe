"""Sequential case-local investigations using ReMe Jobs and sandboxed command tools."""

import json
import sys
from pathlib import Path

from agentscope.tool import FunctionTool, Toolkit
from pydantic import Field, field_validator, model_validator
import yaml

from reme.schema.token_usage import TokenUsage
from reme.steps.base_step import BaseStep, Ref

from ..auto_bash import AutoBash
from ..code.repository import resolve_revision, validate_repository
from ..evaluation.artifacts import contained
from ..evaluation.component import blocking
from ..evaluation.evidence_step import _source_case
from ..evaluation.inputs import entity_id
from ..evaluation.memory_inspection_step import _repository
from ..preparation.spec import SpecModel
from ..workspace import HomeWorkspace
from .sandbox import create_case_study_sandbox
from .wrapper import BoundedAgentWrapper

PROMPT = Path(__file__).with_suffix(".yaml")
DEFAULT_FOCUS = "Investigate the failed or low-scoring queries and their earliest supported causes."
CASE_TOOLS = (
    "home_inspect_validation",
    "home_trace_query_evidence",
    "home_inspect_memory_history",
    "home_compare_memory",
)
CODE_TOOLS = ("home_list_code_tree", "home_search_code", "home_read_code_file")
_SANDBOX_PROBE = """\
import json
import pathlib
import subprocess
import sys

scope = json.loads(sys.argv[1])
with open(scope['source_case'], 'rb') as stream:
    stream.read(1)
with open(pathlib.Path(scope['validation_path']) / 'manifest.json', 'rb') as stream:
    stream.read(1)
subprocess.run([
    '/bin/bash', '--noprofile', '--norc', '-c',
    'git -C "$1" --no-pager -c core.fsmonitor=false -c log.showSignature=false '
    'show --no-ext-diff --no-textconv --no-patch --format=%H "$2"',
    'probe', scope['code_repository'], scope['candidate_commit'],
], check=True)
"""


async def probe_sandbox(backend, scope):
    """Verify the same interpreter, shell, Git and evidence access used by the investigation."""
    return await backend.exec_shell(
        [sys.executable, "-I", "-B", "-c", _SANDBOX_PROBE, json.dumps(scope)],
        cwd=scope["case_path"],
        timeout=10,
    )


class CaseStudyInput(SpecModel):
    """Select one existing validation case; runtime and permissions are host-owned."""

    validation_path: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    focus: str = Field(
        default=DEFAULT_FOCUS,
        min_length=1,
    )

    @field_validator("case_id")
    @classmethod
    def validate_case(cls, value):
        """Keep the case inside its manifest-selected artifact directory."""
        return entity_id(value)


class CaseStudyBatchInput(SpecModel):
    """Share one evaluation and focus across an ordered, bounded case selection."""

    evaluation_path: str = Field(min_length=1)
    case_ids: list[str] = Field(min_length=1, max_length=8)
    focus: str = Field(default=DEFAULT_FOCUS, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy(cls, value):
        """Accept old single-case arguments without accepting ambiguous combinations."""
        if not isinstance(value, dict):
            return value
        value = dict(value)
        for old, new in (("validation_path", "evaluation_path"), ("case_id", "case_ids")):
            if old in value:
                if new in value:
                    raise ValueError(f"Provide {new} or {old}, not both")
                item = value.pop(old)
                value[new] = [item] if old == "case_id" else item
        return value

    @field_validator("case_ids")
    @classmethod
    def validate_cases(cls, values):
        """Reject unsafe or duplicate selectors before any agent runs."""
        for value in values:
            entity_id(value)
        if len(set(values)) != len(values):
            raise ValueError("case_ids must not contain duplicates")
        return values


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

        Read case JSON, memory files and documentation here, not with home_read_code_file.
        For example code="import json; print(json.load(open('/allowed/result.json'))['queries'][0])".
        In code mode do not pass args; in file mode pass file and args, without code.
        Check returncode and stderr: a denied or failed read is not evidence that content is absent.

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
        tool = wrapper._make_tool(  # pylint: disable=protected-access
            job,
            injected_job_kwargs=injected,
        )
        if name in CODE_TOOLS:
            tool.description += (
                " Reads candidate source at the host-pinned evaluated commit, not artifact or documentation files."
                " Paths are repository-relative, for example plugins/beam/src/reme_beam/agentic_answer.yaml;"
                " omit path to search/list the root, never use '.' or an absolute path."
                " Read memory, logs, JSON and home/docs with AutoBash or python_execute instead."
            )
        if "path" in tool.input_schema.get("properties", {}):
            description = (
                "Required candidate-repository-relative file path, never absolute."
                if name == "home_read_code_file"
                else "Literal repository-relative path, never absolute; omit for the whole repository, not '' or '.'."
            )
            tool.input_schema["properties"]["path"] = {
                **tool.input_schema["properties"]["path"],
                "description": description,
            }
        if name == "home_trace_query_evidence":
            tool.description += (
                " For a selected query, start here. locations are JSON Pointers into paths.source_case/query_result;"
                " session source_pointer locates messages in source_case. criteria previews saved rubric judgments"
                " when available. Hit counts and memory coordinates do not prove what reached the answerer."
            )
        if name == "home_inspect_memory_history":
            tool.description += " path filters files within the memory repository; omit it to list all checkpoints."
        if name == "home_compare_memory":
            tool.input_schema["required"] = list(dict.fromkeys([*tool.input_schema.get("required", []), "base_ref"]))
            tool.input_schema["properties"]["base_ref"] = {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Required earlier memory commit from home_inspect_memory_history;"
                    " compare with a different target_ref."
                ),
            }
            tool.description += (
                " Same-case comparison only: supply base_ref and target_ref from home_inspect_memory_history."
                ' Example: {"base_ref": "<earlier commit>", "target_ref": "<later commit>", "path": "daily"}.'
                " Identical refs have no changes; omit path for the whole memory tree."
            )
        tools.append(tool)
    bash = AutoBash(cwd=scope["case_path"], backend=backend, is_read_only=False)
    bash.description = (
        "Read case artifacts, memory, logs and home/docs with shell commands inside the OS file sandbox."
        " No command whitelist; host paths are read-only and network is denied."
        " Use absolute allowed paths, e.g. cat '<case_path>/queries/<query_id>/answer.log', or"
        " git -C '<memory_repository>' show '<commit>:daily/note.md'."
        " Prefer pinned code tools for evaluated candidate source. Keep stderr visible; for pipelines use"
        " set -o pipefail so a failed read is not hidden by head/grep. A failed read does not prove absence."
        " timeout is in milliseconds."
    )
    tools.extend(
        [
            bash,
            python_tool(backend, scope["case_path"]),
        ],
    )
    for tool in tools:
        tool.is_concurrency_safe = False
    return Toolkit(tools=tools)


class HomeCaseStudyStep(BaseStep):
    """Run ordered cases with fresh agents through the normal ReMe lifecycle."""

    workspace: HomeWorkspace = Ref(HomeWorkspace, "home.workspace")
    agent_wrapper: BoundedAgentWrapper = Ref(BoundedAgentWrapper, "agent_wrapper")

    async def execute(self):
        """Validate the whole batch, then isolate each sequential investigation and failure."""
        if self.context.get("_allowed_paths") is not None:
            raise ValueError("Home case study requires unrestricted access to its managed artifacts")
        if not self.workspace.is_started or not self.agent_wrapper.is_started:
            raise RuntimeError("Start the Home workspace and case-study agent wrapper first")
        fields = (*CaseStudyBatchInput.model_fields, "validation_path", "case_id")
        spec = CaseStudyBatchInput.model_validate({key: self.context[key] for key in fields if key in self.context})
        scopes = []
        for case_id in spec.case_ids:
            scopes.append(
                await blocking(
                    prepare_scope,
                    self.workspace.home_path,
                    CaseStudyInput(validation_path=spec.evaluation_path, case_id=case_id, focus=spec.focus),
                ),
            )
        if len({scope["candidate_commit"] for scope in scopes}) != 1:
            raise ValueError("Evaluation candidate commit changed while preparing case scopes")
        output_limit = self.kwargs.get("output_token_limit", 2500)
        if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit < 1:
            raise ValueError("output_token_limit must be a positive integer")
        prompts = yaml.safe_load(PROMPT.read_text(encoding="utf-8"))
        if not isinstance(prompts, dict) or any(
            not isinstance(prompts.get(key), str) or not prompts[key].strip()
            for key in ("system_prompt", "task_prompt")
        ):
            raise ValueError("Case-study YAML requires nonempty system_prompt and task_prompt strings")
        response = self.context.response
        response.metadata.update(
            evaluation_path=scopes[0]["validation_path"],
            candidate_commit=scopes[0]["candidate_commit"],
        )
        # Preserve the original text response and error handling for singular callers.
        if "case_id" in self.context:
            result = await self._investigate(scopes[0], spec.focus, prompts, output_limit)
            response.answer = result["result"]
            response.metadata.update(validation_path=scopes[0]["validation_path"], case_id=spec.case_ids[0])
            response.metadata.update({key: result[key] for key in ("session_id", "usage") if key in result})
            return response
        results = []
        response.answer = {"results": results}
        for scope in scopes:
            try:
                result = await self._investigate(scope, spec.focus, prompts, output_limit)
                results.append(
                    {
                        "case_id": scope["case_id"],
                        "success": True,
                        "report": result["result"],
                        **{key: result[key] for key in ("session_id", "usage") if key in result},
                    },
                )
            except Exception as exc:  # A failed investigator must not discard or cancel sibling reports.
                self.logger.exception(f"Case-study investigation failed: case_id={scope['case_id']}")
                results.append({"case_id": scope["case_id"], "success": False, "error": f"{type(exc).__name__}: {exc}"})
        response.success = all(item["success"] for item in results)
        # Failed or unreported calls have unknown usage; never present a partial sum as a full total.
        response.metadata["usage"] = (
            TokenUsage.combine([TokenUsage.model_validate(item["usage"]) for item in results]).model_dump()
            if all(item.get("usage") is not None for item in results)
            else None
        )
        return response

    async def _investigate(self, scope, focus, prompts, output_limit):
        """Build a fresh case-scoped toolkit and SDK agent for exactly one case."""
        backend = create_case_study_sandbox(
            cwd=scope["case_path"],
            read_only_paths=scope["read_only_paths"],
            writable_paths=[],
            memory_limit_mib=self.kwargs.get("memory_limit_mib"),
        )
        probe = await probe_sandbox(backend, scope)
        if not probe.ok():
            raise RuntimeError("Case-study sandbox unavailable: " + probe.stderr.decode(errors="replace"))
        toolkit = create_toolkit(self.agent_wrapper, scope, backend)
        prompt = prompts["system_prompt"].format(**scope, output_token_limit=output_limit)
        return await self.agent_wrapper.reply(
            prompts["task_prompt"].format(focus=focus),
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
