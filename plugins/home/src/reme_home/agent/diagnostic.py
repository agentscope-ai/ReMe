"""Diagnose saved Home evaluations with an isolated evidence-driven agent."""

import asyncio
import copy
import json

from agentscope.state import AgentState
from agentscope.tool import FunctionTool
from pydantic import Field, field_validator

from reme.steps.base_step import BaseStep, Ref

from ..evaluation.artifacts import contained, read_json, write_json
from ..evaluation.component import blocking
from ..preparation.spec import SpecModel
from ..workspace import HomeWorkspace
from .workflow_common import required_job, save_record, workflow_reply
from .wrapper import BoundedAgentWrapper


class DiagnosticInput(SpecModel):
    """Require explicit evaluation evidence; caller state is never model-selected."""

    evaluation_paths: list[str] = Field(min_length=1)
    instruction: str = Field(default="Identify and rank the evidence-supported root causes.", min_length=1)

    @field_validator("evaluation_paths")
    @classmethod
    def validate_paths(cls, values):
        """Reject empty and duplicate input references before creating an agent."""
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("evaluation_paths must contain unique nonempty paths")
        return values


def evaluation_paths(home, values):
    """Require explicit, unique local runs with actual evidence, including failed runs."""
    if not values:
        raise ValueError("evaluation_paths must identify at least one run")
    paths = []
    root = contained(home, "evaluations", directory=True)
    for value in values:
        path = contained(home, value, directory=True)
        if not path.is_relative_to(root) or path == root:
            raise ValueError("evaluation_paths must be inside home/evaluations")
        read_json(contained(path, "manifest.json"))
        evidence = [path / "summary.json", path / "failure.json"]
        evidence += list(path.glob("cases/*/memory_construction/result.json"))
        evidence += list(path.glob("cases/*/queries/result.json"))
        if not any(contained(path, item).is_file() for item in evidence):
            raise ValueError(f"Evaluation contains no results: {path}")
        paths.append(str(path))
    if len(paths) != len(set(paths)):
        raise ValueError("evaluation_paths must be unique")
    return paths


def trajectory(state):
    """Serialize only caller messages and compression summary, never credentials or permissions."""
    if state is not None and not isinstance(state, AgentState):
        raise TypeError("Injected caller state must be an AgentState")
    if state is None:
        return {"messages": [], "summary": None}
    summary = state.summary
    if isinstance(summary, list):
        summary = [block.model_dump(mode="json") for block in summary]
    return {
        "session_id": state.session_id,
        "messages": [message.model_dump(mode="json") for message in state.context],
        "summary": summary,
    }


def diagnostic_tool(wrapper):
    """Inject the optimizer's real trajectory into a fresh diagnostic invocation."""
    job = required_job(wrapper, "home_diagnose")

    async def diagnose(
        evaluation_paths: list[str],  # pylint: disable=redefined-outer-name
        instruction: str = "Identify and rank the evidence-supported root causes.",
        **kwargs,
    ):
        if set(kwargs) - {"_agent_state"}:
            raise ValueError("Caller trajectory is host-owned and cannot be supplied by the model")
        response = await job(
            instruction=instruction,
            evaluation_paths=evaluation_paths,
            caller_trajectory=trajectory(kwargs.get("_agent_state")),
        )
        return {"success": response.success, "answer": response.answer, "metadata": response.metadata}

    tool = FunctionTool(
        diagnose,
        name="home_diagnose",
        description=job.description,
        is_concurrency_safe=False,
        is_state_injected=True,
    )
    tool.input_schema = copy.deepcopy(job.parameters)
    return tool


class HomeDiagnoseStep(BaseStep):
    """Aggregate explicit evaluation evidence through a read-only diagnostic agent."""

    workspace: HomeWorkspace = Ref(HomeWorkspace, "home.workspace")
    agent_wrapper: BoundedAgentWrapper = Ref(BoundedAgentWrapper, "agent_wrapper")

    async def execute(self):
        """Validate evidence, save the host trajectory and persist a provenance-bearing report."""
        if self.context.get("_allowed_paths") is not None:
            raise ValueError("Home diagnosis requires access to its managed evidence")
        if not self.workspace.is_started or not self.agent_wrapper.is_started:
            raise RuntimeError("Start the Home workspace and diagnostic wrapper first")
        spec = DiagnosticInput.model_validate(
            {key: self.context[key] for key in DiagnosticInput.model_fields if key in self.context},
        )
        home = self.workspace.home_path
        paths = await blocking(evaluation_paths, home, spec.evaluation_paths)
        caller = self.context.get("caller_trajectory", trajectory(None))
        if not isinstance(caller, dict):
            raise ValueError("Caller trajectory must be a host-supplied object")
        source = await blocking(save_record, home, "diagnostic-trajectories", caller)
        payload = {
            "status": "running",
            "instruction": spec.instruction,
            "evaluation_paths": paths,
            "trajectory_path": source,
        }
        report_path = await blocking(save_record, home, "diagnostic-reports", payload)
        self.context.response.metadata.update(diagnostic_report_path=report_path, trajectory_path=source)
        try:
            result = await workflow_reply(
                self.agent_wrapper,
                home,
                "diagnostic",
                f"Diagnostic request:\n{spec.instruction}\n\nComplete available caller trajectory: {source}\n"
                f"Explicit evaluation paths: {json.dumps(paths, ensure_ascii=False)}",
                job_names=("home_case_study",),
                memory_limit_mib=self.kwargs.get("memory_limit_mib"),
            )
            if not isinstance(result.get("result"), str) or not result["result"].strip():
                raise ValueError("Diagnostic agent returned no report")
            payload.update(
                status="completed",
                diagnosis=result["result"],
                session_id=result.get("session_id"),
                usage=result.get("usage"),
            )
            self.context.response.answer = {**payload, "diagnostic_report_path": report_path}
            self.context.response.metadata.update(
                {key: result[key] for key in ("session_id", "usage") if key in result},
            )
        except BaseException as exc:
            payload.update(status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed", error=str(exc))
            raise
        finally:
            await blocking(write_json, contained(home, report_path), payload)
        return self.context.response
