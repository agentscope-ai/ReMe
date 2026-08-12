"""AgentScope tool that invokes the Meta-ReMe diagnostic subagent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from agentscope.model import ChatModelBase
from agentscope.state import AgentState
from agentscope.tool import FunctionTool

from ..agent.diagnostic_agent import create_diagnostic_agent, message_text, run_diagnostic_agent
from ..utils import resolve_workspace, serialize_trajectory


def create_diagnostic_subagent_tool(
    model: ChatModelBase,
    workspace: str | Path | None = None,
    *,
    max_iters: int = 30,
) -> FunctionTool:
    """Create a state-injected tool that delegates diagnosis to a fresh subagent."""

    workspace_path = resolve_workspace(workspace)

    async def diagnose_code(
        instruction: str = "",
        validation_paths: list[str] | None = None,
        trajectory: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Diagnose the current candidate using Git, validations, and trajectory.

        Args:
            instruction: Specific question or suspected regression to investigate.
            validation_paths: Validation details_path values that prompted diagnosis.
            trajectory: Optional extra runtime trajectory not already in caller state.
        """

        paths = _validation_paths(workspace_path, validation_paths)
        caller_state = kwargs.get("_agent_state")
        if caller_state is not None and not isinstance(caller_state, AgentState):
            raise TypeError("_agent_state must be an AgentState")
        trajectory_path = _write_trajectory(workspace_path, serialize_trajectory(caller_state, trajectory))
        agent = create_diagnostic_agent(model, workspace_path, max_iters=max_iters)
        reply = await run_diagnostic_agent(
            agent,
            request=instruction,
            trajectory_path=trajectory_path,
            validation_paths=paths,
        )
        return {
            "status": "completed",
            "diagnosis": message_text(reply),
            "trajectory_path": str(trajectory_path),
            "validation_paths": [str(path) for path in paths],
        }

    tool = FunctionTool(
        diagnose_code,
        name="diagnostic_subagent_tool",
        description=(
            "Invoke a fresh read-only diagnostic subagent after validation. It automatically receives the caller's "
            "full AgentScope trajectory and can inspect all validation artifacts, sandbox memory evolution, Git "
            "history, code diffs, and cross-version scores."
        ),
        is_concurrency_safe=False,
        is_read_only=False,
        is_state_injected=True,
    )
    # AgentScope injects this through **kwargs; it is intentionally absent from
    # the model-visible schema so callers cannot spoof another agent's state.
    return tool


def _validation_paths(workspace: Path, values: list[str] | None) -> list[Path]:
    if values is None:
        return []
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("validation_paths must be a list of non-empty strings or null")
    resolved: list[Path] = []
    evaluations = (workspace / "evaluations").resolve()
    for value in values:
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (workspace / path).resolve()
        try:
            path.relative_to(evaluations)
        except ValueError as exc:
            raise ValueError(f"validation path is outside this workspace's evaluations: {value}") from exc
        if not path.is_dir() or not (path / "manifest.json").is_file():
            raise ValueError(f"validation path is not a result directory: {value}")
        resolved.append(path)
    if len(resolved) != len(set(resolved)):
        raise ValueError("validation_paths must be unique")
    return resolved


def _write_trajectory(workspace: Path, payload: dict[str, Any]) -> Path:
    directory = workspace / "logs/diagnostic-trajectories"
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".trajectory-", suffix=".json", dir=directory)
    temporary = Path(temporary_name)
    destination = directory / f"trajectory-{temporary.stem.removeprefix('.trajectory-')}.json"
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return destination
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
