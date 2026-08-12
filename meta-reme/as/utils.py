"""Shared construction helpers for Meta-ReMe AgentScope agents.

Agent implementation stays in Python while all behavioral instructions live
next to the implementation in a same-stem YAML file.  This module also owns
the small, read-only inspection tools shared by diagnostic entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from agentscope.state import AgentState
from agentscope.tool import FunctionTool
import yaml

from runtime import TOOL_RUNTIME

CODE_REPOSITORY = Path("code/repo/reme")
MAX_GIT_OUTPUT_CHARACTERS = 200_000


@dataclass(frozen=True)
class ContextCompressionConfig:
    """Agent-local context compression settings loaded from its YAML companion."""

    enabled: bool = False
    trigger_ratio: float = 0.8
    reserve_ratio: float = 0.1


@dataclass(frozen=True)
class AgentPrompt:
    """Validated prompt text loaded from one agent's YAML companion."""

    name: str
    model: str
    system_prompt: str
    task_prompt: str
    context_compression: ContextCompressionConfig

    def render_system(self, **values: object) -> str:
        """Render the system prompt with trusted runtime paths."""

        return self.system_prompt.format_map(values)

    def render_task(self, **values: object) -> str:
        """Render the initial task prompt with trusted runtime values."""

        return self.task_prompt.format_map(values)


def load_agent_prompt(agent_file: str | Path) -> AgentPrompt:
    """Load ``<agent_file stem>.yaml`` and validate its prompt fields."""

    agent_file = Path(agent_file).resolve()
    prompt_file = agent_file.with_suffix(".yaml")
    payload = yaml.safe_load(prompt_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Agent prompt must be a YAML mapping: {prompt_file}")
    unknown = set(payload) - {"name", "model", "system_prompt", "task_prompt", "context_compression"}
    if unknown:
        raise ValueError(f"Unknown agent prompt fields in {prompt_file}: {sorted(unknown)}")
    values: dict[str, str] = {}
    for field in ("name", "model", "system_prompt", "task_prompt"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Agent prompt field {field!r} must be a non-empty string: {prompt_file}")
        values[field] = value.strip()
    compression = _load_context_compression(payload.get("context_compression", {}), prompt_file)
    return AgentPrompt(**values, context_compression=compression)


def _load_context_compression(value: Any, prompt_file: Path) -> ContextCompressionConfig:
    """Validate optional per-agent context compression settings."""

    if not isinstance(value, dict):
        raise ValueError(f"context_compression must be a YAML mapping: {prompt_file}")
    unknown = set(value) - {"enabled", "trigger_ratio", "reserve_ratio"}
    if unknown:
        raise ValueError(f"Unknown context compression fields in {prompt_file}: {sorted(unknown)}")
    enabled = value.get("enabled", False)
    trigger_ratio = value.get("trigger_ratio", 0.8)
    reserve_ratio = value.get("reserve_ratio", 0.1)
    if not isinstance(enabled, bool):
        raise ValueError(f"context_compression.enabled must be a boolean: {prompt_file}")
    for field, ratio in (("trigger_ratio", trigger_ratio), ("reserve_ratio", reserve_ratio)):
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 < ratio < 0.9:
            raise ValueError(f"context_compression.{field} must be a number between 0 and 0.9: {prompt_file}")
    if reserve_ratio >= trigger_ratio:
        raise ValueError(f"context_compression.reserve_ratio must be less than trigger_ratio: {prompt_file}")
    return ContextCompressionConfig(enabled, float(trigger_ratio), float(reserve_ratio))


def resolve_workspace(workspace: str | Path | None = None) -> Path:
    """Resolve an explicit workspace or the process-local tool runtime."""

    if workspace is None:
        workspace, _ = TOOL_RUNTIME.require_configured()
    resolved = Path(workspace).resolve()
    if not (resolved / ".meta-reme-workspace.json").is_file():
        raise ValueError(f"Not a prepared Meta-ReMe workspace: {resolved}")
    repository = resolved / CODE_REPOSITORY
    if not repository.is_dir() or not (repository / ".git").exists():
        raise ValueError(f"Meta-ReMe candidate repository is missing: {repository}")
    return resolved


def serialize_trajectory(state: AgentState | None, extra: str | None = None) -> dict[str, Any]:
    """Return a lossless JSON-compatible snapshot of the caller trajectory."""

    return {
        "agent_state": state.model_dump(mode="json", exclude_none=False) if state is not None else None,
        "extra_trajectory": extra,
    }


def validation_catalog(workspace: Path) -> dict[str, Any]:
    """Index every published validation without hiding detailed artifact paths."""

    evaluations = workspace / "evaluations"
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(evaluations.glob("*/*/*/manifest.json")):
        result_dir = manifest_path.parent
        relative_parts = result_dir.relative_to(evaluations).parts
        if len(relative_parts) != 3:
            continue
        branch, commit, validation_id = relative_parts
        entry: dict[str, Any] = {
            "branch": branch,
            "commit": commit,
            "validation_id": validation_id,
            "details_path": str(result_dir),
            "manifest_path": str(manifest_path),
        }
        for name in ("summary.json", "failure.json"):
            path = result_dir / name
            if path.is_file():
                entry[name.removesuffix(".json")] = _read_json(path)
                entry[f"{name.removesuffix('.json')}_path"] = str(path)
        entries.append(entry)
    return {"workspace": str(workspace), "validation_count": len(entries), "validations": entries}


def create_validation_catalog_tool(workspace: Path) -> FunctionTool:
    """Create a read-only tool that lists all validation runs and summaries."""

    workspace = resolve_workspace(workspace)

    def list_validation_results() -> dict[str, Any]:
        """List every validation and its summary/failure plus full artifact path."""

        return validation_catalog(workspace)

    return FunctionTool(
        list_validation_results,
        name="list_validation_results",
        description=(
            "List all validation runs across branches and commits, including scores, failures, "
            "and paths to every run's complete artifacts."
        ),
        is_read_only=True,
    )


def create_git_inspection_tools(workspace: Path) -> list[FunctionTool]:
    """Create bounded, read-only Git history and diff tools for one workspace."""

    repository = resolve_workspace(workspace) / CODE_REPOSITORY

    def inspect_git_history(max_count: int = 50, path: str | None = None) -> dict[str, Any]:
        """Inspect candidate commits, branches, decorations, and optional path history.

        Args:
            max_count: Maximum number of commits to return, from 1 through 200.
            path: Optional repository-relative path whose history should be followed.
        """

        if not isinstance(max_count, int) or isinstance(max_count, bool) or not 1 <= max_count <= 200:
            raise ValueError("max_count must be an integer from 1 through 200")
        arguments = [
            "log",
            "--all",
            f"--max-count={max_count}",
            "--date=iso-strict",
            "--pretty=format:%H%x09%P%x09%D%x09%aI%x09%s",
        ]
        if path is not None:
            arguments.extend(["--", _safe_repository_path(repository, path)])
        return {
            "repository": str(repository),
            "status": _git(repository, "status", "--short", "--branch"),
            "branches": _git(repository, "branch", "--all", "--verbose", "--no-abbrev"),
            "history": _git(repository, *arguments),
        }

    def compare_git_versions(base_ref: str, target_ref: str, path: str | None = None) -> dict[str, Any]:
        """Compare two candidate revisions and return their code diff and statistics.

        Args:
            base_ref: Existing base revision, branch, tag, or full commit SHA.
            target_ref: Existing target revision, branch, tag, or full commit SHA.
            path: Optional repository-relative path used to narrow the comparison.
        """

        base_sha = _resolve_revision(repository, base_ref)
        target_sha = _resolve_revision(repository, target_ref)
        path_arguments = ["--", _safe_repository_path(repository, path)] if path is not None else []
        return {
            "repository": str(repository),
            "base_commit": base_sha,
            "target_commit": target_sha,
            "stat": _git(repository, "diff", "--stat", base_sha, target_sha, *path_arguments),
            "diff": _git(repository, "diff", "--no-ext-diff", base_sha, target_sha, *path_arguments),
        }

    return [
        FunctionTool(inspect_git_history, is_read_only=True),
        FunctionTool(compare_git_versions, is_read_only=True),
    ]


def create_memory_inspection_tools(workspace: Path) -> list[FunctionTool]:
    """Create bounded tools for Git history stored in validation memory artifacts."""

    workspace = resolve_workspace(workspace)

    def list_memory_histories(validation_path: str | None = None) -> dict[str, Any]:
        """List case memory repositories and their checkpoint counts.

        Args:
            validation_path: Optional validation details_path. Omit to inspect every validation.
        """

        validations = (
            [_resolve_validation_result(workspace, validation_path)]
            if validation_path is not None
            else [path.parent for path in sorted((workspace / "evaluations").glob("*/*/*/manifest.json"))]
        )
        histories: list[dict[str, Any]] = []
        for validation in validations:
            for case_path in sorted((validation / "cases").iterdir()) if (validation / "cases").is_dir() else []:
                if not case_path.is_dir():
                    continue
                repository = case_path / "memory_construction/reme_workspace"
                if not (repository / ".git").exists():
                    histories.append(
                        {
                            "validation_path": str(validation),
                            "case_id": case_path.name,
                            "repository": str(repository),
                            "has_git_history": False,
                        },
                    )
                    continue
                try:
                    head = _git(repository, "rev-parse", "--verify", "HEAD^{commit}").strip()
                except RuntimeError:
                    histories.append(
                        {
                            "validation_path": str(validation),
                            "case_id": case_path.name,
                            "repository": str(repository),
                            "has_git_history": False,
                            "checkpoint_count": 0,
                        },
                    )
                    continue
                count = int(_git(repository, "rev-list", "--count", head).strip() or "0")
                histories.append(
                    {
                        "validation_path": str(validation),
                        "case_id": case_path.name,
                        "repository": str(repository),
                        "has_git_history": True,
                        "head": head,
                        "checkpoint_count": count,
                        "latest_subject": _git(repository, "log", "-1", "--pretty=%s").strip(),
                    },
                )
        return {"workspace": str(workspace), "history_count": len(histories), "histories": histories}

    def inspect_memory_history(
        validation_path: str,
        case_id: str,
        max_count: int = 100,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Inspect session checkpoints in one exported case memory repository.

        Args:
            validation_path: Validation details_path from list_validation_results.
            case_id: Case ID within that validation.
            max_count: Maximum number of commits, from 1 through 200.
            path: Optional memory-workspace-relative path used to follow one file or directory.
        """

        if not isinstance(max_count, int) or isinstance(max_count, bool) or not 1 <= max_count <= 200:
            raise ValueError("max_count must be an integer from 1 through 200")
        repository = _memory_repository(workspace, validation_path, case_id)
        path_arguments = ["--", _safe_repository_path(repository, path)] if path is not None else []
        return {
            "validation_path": str(_resolve_validation_result(workspace, validation_path)),
            "case_id": case_id,
            "repository": str(repository),
            "status": _git(repository, "status", "--short", "--branch"),
            "history": _git(
                repository,
                "log",
                f"--max-count={max_count}",
                "--date=iso-strict",
                "--stat",
                "--pretty=format:commit %H%nparents %P%ndate %aI%nsubject %s",
                *path_arguments,
            ),
        }

    def compare_memory_versions(
        validation_path: str,
        case_id: str,
        base_ref: str,
        target_ref: str,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Compare two session checkpoints from one case's memory history.

        Args:
            validation_path: Validation details_path from list_validation_results.
            case_id: Case ID within that validation.
            base_ref: Earlier memory commit SHA or revision.
            target_ref: Later memory commit SHA or revision.
            path: Optional memory-workspace-relative path used to narrow the diff.
        """

        repository = _memory_repository(workspace, validation_path, case_id)
        base_sha = _resolve_revision(repository, base_ref)
        target_sha = _resolve_revision(repository, target_ref)
        path_arguments = ["--", _safe_repository_path(repository, path)] if path is not None else []
        return {
            "validation_path": str(_resolve_validation_result(workspace, validation_path)),
            "case_id": case_id,
            "repository": str(repository),
            "base_commit": base_sha,
            "target_commit": target_sha,
            "stat": _git(repository, "diff", "--stat", base_sha, target_sha, *path_arguments),
            "diff": _git(repository, "diff", "--no-ext-diff", base_sha, target_sha, *path_arguments),
        }

    def compare_memory_snapshots(
        base_validation_path: str,
        target_validation_path: str,
        case_id: str,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Compare final tracked memory files for one case across code validations.

        Args:
            base_validation_path: Earlier validation details_path.
            target_validation_path: Later validation details_path.
            case_id: Case ID present in both validations.
            path: Optional memory-workspace-relative file or directory.
        """

        base_repository = _memory_repository(workspace, base_validation_path, case_id)
        target_repository = _memory_repository(workspace, target_validation_path, case_id)
        safe_path = _safe_repository_path(base_repository, path) if path is not None else None
        if safe_path is not None:
            _safe_repository_path(target_repository, path)
        base_files = _tracked_memory_files(base_repository, safe_path)
        target_files = _tracked_memory_files(target_repository, safe_path)
        names = sorted(set(base_files) | set(target_files))
        added = [name for name in names if name not in base_files]
        removed = [name for name in names if name not in target_files]
        modified = [
            name
            for name in names
            if name in base_files and name in target_files and base_files[name] != target_files[name]
        ]
        diff_parts: list[str] = []
        for name in added + removed + modified:
            before = base_files.get(name, b"")
            after = target_files.get(name, b"")
            try:
                before_lines = before.decode("utf-8").splitlines(keepends=True)
                after_lines = after.decode("utf-8").splitlines(keepends=True)
            except UnicodeDecodeError:
                diff_parts.append(f"Binary files differ: {name}\n")
                continue
            diff_parts.extend(
                difflib.unified_diff(
                    before_lines,
                    after_lines,
                    fromfile=f"base/{name}",
                    tofile=f"target/{name}",
                ),
            )
            if sum(len(part) for part in diff_parts) > MAX_GIT_OUTPUT_CHARACTERS:
                diff_parts.append("[output truncated; narrow the path]\n")
                break
        return {
            "base_validation_path": str(_resolve_validation_result(workspace, base_validation_path)),
            "target_validation_path": str(_resolve_validation_result(workspace, target_validation_path)),
            "case_id": case_id,
            "path": safe_path,
            "added": added,
            "removed": removed,
            "modified": modified,
            "base_tree_sha256": _memory_tree_hash(base_files),
            "target_tree_sha256": _memory_tree_hash(target_files),
            "diff": "".join(diff_parts)[:MAX_GIT_OUTPUT_CHARACTERS],
        }

    return [
        FunctionTool(list_memory_histories, is_read_only=True),
        FunctionTool(inspect_memory_history, is_read_only=True),
        FunctionTool(compare_memory_versions, is_read_only=True),
        FunctionTool(compare_memory_snapshots, is_read_only=True),
    ]


def _resolve_validation_result(workspace: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("validation_path must be a non-empty string")
    supplied = Path(value)
    resolved = supplied.resolve() if supplied.is_absolute() else (workspace / supplied).resolve()
    evaluations = (workspace / "evaluations").resolve()
    try:
        relative = resolved.relative_to(evaluations)
    except ValueError as exc:
        raise ValueError(f"validation_path is outside this workspace: {value}") from exc
    if len(relative.parts) != 3 or not (resolved / "manifest.json").is_file():
        raise ValueError(f"validation_path is not a published validation result: {value}")
    return resolved


def _memory_repository(workspace: Path, validation_path: str, case_id: str) -> Path:
    validation = _resolve_validation_result(workspace, validation_path)
    if not isinstance(case_id, str) or not case_id or Path(case_id).name != case_id or case_id in {".", ".."}:
        raise ValueError("case_id must be one safe path segment")
    repository = validation / "cases" / case_id / "memory_construction/reme_workspace"
    if not repository.is_dir() or not (repository / ".git").exists():
        raise ValueError(f"case has no exported memory Git repository: {case_id}")
    return repository.resolve()


def _tracked_memory_files(repository: Path, path: str | None) -> dict[str, bytes]:
    arguments = ["ls-files", "-z"]
    if path is not None:
        arguments.extend(["--", path])
    names = [name for name in _git(repository, *arguments).split("\0") if name]
    if len(names) > 10_000:
        raise ValueError("memory snapshot has more than 10,000 tracked files; narrow the path")
    files: dict[str, bytes] = {}
    for name in names:
        safe_name = _safe_repository_path(repository, name)
        value = (repository / safe_name).read_bytes()
        if len(value) > 5_000_000:
            raise ValueError(f"tracked memory file exceeds 5 MB; narrow the path: {safe_name}")
        files[safe_name] = value
    return files


def _memory_tree_hash(files: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(value).digest())
    return digest.hexdigest()


def _resolve_revision(repository: Path, revision: str) -> str:
    if not isinstance(revision, str) or not revision or revision.startswith("-"):
        raise ValueError("Git revisions must be non-empty and may not start with '-'")
    return _git(repository, "rev-parse", "--verify", f"{revision}^{{commit}}").strip()


def _safe_repository_path(repository: Path, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be repository-relative: {value}")
    resolved = (repository / path).resolve()
    try:
        relative = resolved.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"path escapes candidate repository: {value}") from exc
    return relative.as_posix()


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    output = result.stdout
    if len(output) > MAX_GIT_OUTPUT_CHARACTERS:
        return output[:MAX_GIT_OUTPUT_CHARACTERS] + "\n[output truncated; narrow the path or refs]"
    return output


def _read_json(path: Path) -> Mapping[str, Any] | list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"Expected JSON object or array: {path}")
    return payload
