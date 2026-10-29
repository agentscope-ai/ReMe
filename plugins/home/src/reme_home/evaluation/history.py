"""Successful evaluation history and rebuildable sampling evidence owned by Home."""

from pathlib import Path
from typing import Mapping, Literal
import math

from pydantic import Field

from reme.components.base_component import BaseComponent

from ..preparation.data.models import utc_now
from ..preparation.spec import SpecModel
from ..sampling import SamplingPolicy, _empty_record, _observe, _query_key
from ..workspace import HomeWorkspace, _initialization_lock, _validate_directory
from .artifacts import contained, read_json, write_json


class EvaluationRecord(SpecModel):
    """A compact catalog entry backed by an immutable successful result directory."""

    run_key: str
    branch_name: str
    commit_sha: str
    validation_id: str
    round: int
    case_ids: list[str]
    query_ids: dict[str, list[str]]
    execution_mode: str
    finished_at: str
    case_count: int
    candidate_failure_count: int
    scored_query_count: int
    mean_query_score: float | None


class HistoryIndex(SpecModel):
    """A disposable index; manifests, summaries and plans remain authoritative."""

    schema_version: Literal[1] = 1
    results: list[EvaluationRecord] = Field(default_factory=list)


def successful_result(home: Path, run: Path) -> tuple[dict, dict] | None:
    """Match the non-debug Response.success contract, including valid low scores."""
    manifest = read_json(contained(home, run / "manifest.json"))
    if manifest.get("options", {}).get("record_history") is False:
        return None
    if (
        manifest.get("mode") != "scored"
        or manifest.get("dataset_kind") != "training-set"
        or manifest.get("status") != "completed"
        or manifest.get("success") is False
        or not manifest.get("sandbox_started_at")
    ):
        return None
    path = contained(home, run / "summary.json")
    if not path.is_file():
        return None
    summary = read_json(path)
    if summary.get("case_status_counts", {}).get("infra_error") != 0 or not summary.get("case_count"):
        return None
    return manifest, summary


def result_record(home: Path, run: Path) -> EvaluationRecord | None:
    """Exclude failed, cancelled, debug and incomplete directories from the catalog."""
    result = successful_result(home, run)
    if result is None:
        return None
    manifest, summary = result
    return EvaluationRecord(
        run_key=run.relative_to(home / "evaluations").as_posix(),
        branch_name=manifest["branch_name"],
        commit_sha=manifest["commit_sha"],
        validation_id=manifest["validation_id"],
        round=manifest["sampling"]["round"],
        case_ids=manifest["case_ids"],
        query_ids=manifest["query_ids"],
        execution_mode=manifest["execution_mode"],
        finished_at=manifest["finished_at"],
        case_count=summary["case_count"],
        candidate_failure_count=summary["case_status_counts"]["candidate_failure"],
        scored_query_count=summary["scored_query_count"],
        mean_query_score=summary["mean_query_score"],
    )


def read_state(home: Path) -> dict:
    """Read a derived cache; started plans and scored artifacts retain its source evidence."""
    path = contained(home, "sampling/state.json")
    if not path.exists():
        return rebuild_sampling_state(home, persist=False)
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        raise ValueError("Unsupported sampling state schema")
    if not isinstance(payload.get("round"), int) or isinstance(payload["round"], bool) or payload["round"] < 0:
        raise ValueError("Invalid sampling round")
    for name in ("cases", "queries", "started_validations"):
        if not isinstance(payload.get(name, {}), dict):
            raise ValueError(f"Invalid sampling {name}")
        payload.setdefault(name, {})
    if not isinstance(payload.get("processed_validations"), list):
        raise ValueError("Invalid processed validations")
    if payload["schema_version"] == 1 or "successful_round" not in payload:
        return rebuild_sampling_state(home, persist=False)
    if (
        not isinstance(payload["successful_round"], int)
        or isinstance(payload["successful_round"], bool)
        or (payload["successful_round"] < 0)
    ):
        raise ValueError("Invalid successful-history round")
    return payload


def start_validation(home: Path, run: Path, policy: SamplingPolicy, plan: dict | None = None) -> dict:
    """Record once after the first successfully initialized sandbox, never before."""
    manifest = read_json(contained(home, run / "manifest.json"))
    if (
        manifest.get("mode") == "debug"
        or manifest.get("dataset_kind") != "training-set"
        or manifest.get("options", {}).get("record_history") is False
    ):
        return {}
    key = run.relative_to(home / "evaluations").as_posix()
    state = read_state(home)
    if key in state["started_validations"]:
        return read_json(contained(home, manifest["sampling"]["plan_path"]))
    plan = (
        dict(plan)
        if plan is not None
        else {
            "schema_version": 1,
            "mode": "explicit",
            "case_ids": manifest["case_ids"],
            "query_ids": manifest["query_ids"],
            "policy": policy.model_dump(),
        }
    )
    # Preserve the draw's state snapshot separately from the committed start round.
    if "round" in plan:
        plan["selection_round"] = plan["round"]
    plan.update(
        round=state["round"] + 1,
        validation_id=manifest["validation_id"],
        validation_key=key,
        sandbox_started_at=utc_now().isoformat(),
    )
    relative = f"sampling/plans/{key}.json"
    write_json(contained(home, relative), plan)
    manifest["sampling"] = {"round": plan["round"], "mode": plan["mode"], "plan_path": relative}
    manifest["sandbox_started_at"] = plan["sandbox_started_at"]
    write_json(run / "manifest.json", manifest)
    state["round"] = plan["round"]
    state["started_validations"][key] = plan["round"]
    write_json(contained(home, "sampling/state.json"), state)
    return plan


def update_sampling_state(home: Path, run: Path, *, _state: dict | None = None) -> None:
    """Update observations only from successful completed formal evaluations."""
    result = successful_result(home, run)
    if result is None:
        return
    manifest, summary = result
    plan = read_json(contained(home, manifest["sampling"]["plan_path"]))
    policy = SamplingPolicy.model_validate(plan["policy"])
    state = read_state(home) if _state is None else _state
    key = plan["validation_key"]
    if key in state["processed_validations"]:
        return
    round_id = state["successful_round"] + 1
    cases, queries = state["cases"], state["queries"]
    summary_cases = summary.get("cases", {})
    items = (
        summary_cases.items()
        if isinstance(summary_cases, Mapping)
        else ((case.get("case_id"), case) for case in summary_cases)
    )
    for case_id, case_summary in items:
        if not isinstance(case_id, str) or not isinstance(case_summary, Mapping):
            continue
        folder = contained(run, f"cases/{case_id}", directory=True)
        query_path = contained(folder, "queries/result.json")
        query_rows = read_json(query_path).get("queries", []) if query_path.is_file() else []
        failure_path = contained(folder, "failure.json")
        failure = read_json(failure_path) if failure_path.is_file() else {}
        # Home summaries omit the redundant case status; reconstruct it using
        # the same authoritative failure/query evidence as summarize().
        status = case_summary.get("status") or (
            "infra_error"
            if failure.get("failure_kind") == "infra_error"
            or any(query.get("infrastructure_error") for query in query_rows)
            else "candidate_failure" if failure or any(query.get("error") for query in query_rows) else "completed"
        )
        case_record = cases.setdefault(case_id, _empty_record())
        case_record["last_selected_round"] = max(case_record.get("last_selected_round") or 0, round_id)
        if status == "infra_error":
            case_record["infra_error_count"] += 1
            continue
        if status == "candidate_failure":
            case_record["candidate_failure_count"] += 1
        scores = []
        for query in query_rows:
            query_id = query.get("query_id")
            if not isinstance(query_id, str):
                continue
            query_record = queries.setdefault(_query_key(case_id, query_id), _empty_record())
            query_record["last_selected_round"] = max(query_record.get("last_selected_round") or 0, round_id)
            if query.get("infrastructure_error"):
                query_record["infra_error_count"] += 1
                continue
            if query.get("failure_kind") == "candidate_failure":
                query_record["candidate_failure_count"] += 1
            score = query.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                scores.append(float(score))
                _observe(query_record, 1.0 - score, round_id, policy.ewma_alpha)
        if scores:
            _observe(case_record, 1.0 - sum(scores) / len(scores), round_id, policy.ewma_alpha)
    state["successful_round"] = round_id
    state["processed_validations"].append(key)
    if _state is None:
        write_json(contained(home, "sampling/state.json"), state)


def rebuild_sampling_state(home: Path, *, persist: bool = True) -> dict:
    """Rebuild the disposable cache from immutable started plans and scored artifacts."""
    state = {
        "schema_version": 2,
        "round": 0,
        "successful_round": 0,
        "cases": {},
        "queries": {},
        "started_validations": {},
        "processed_validations": [],
    }
    runs = []
    for path in contained(home, "evaluations", directory=True).glob("*/*/*/manifest.json"):
        manifest = read_json(contained(home, path))
        if (
            manifest.get("mode") == "debug"
            or manifest.get("dataset_kind") != "training-set"
            or manifest.get("options", {}).get("record_history") is False
            or not (manifest.get("sandbox_started_at"))
        ):
            continue
        plan = read_json(contained(home, manifest["sampling"]["plan_path"]))
        runs.append((plan["round"], path.parent, plan["validation_key"]))
    for round_id, run, key in sorted(runs):
        state["round"] = max(state["round"], round_id)
        state["started_validations"][key] = round_id
        update_sampling_state(home, run, _state=state)
    if persist:
        write_json(contained(home, "sampling/state.json"), state)
    return state


class HomeEvaluationHistory(BaseComponent):
    """Own successful-result lookup, registration and sampling caches for one workspace."""

    component_type = "home.evaluation_history"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workspace = self.bind("default", HomeWorkspace, optional=False)

    def _history_lock(self):
        if not self.is_started:
            raise RuntimeError("Start the Home application before reading evaluation history")
        _validate_directory(self.workspace_path.resolve(), self.workspace.state_path)
        return _initialization_lock(self.workspace.state_path / "workspace.lock")

    def _run_path(self, run: Path) -> Path:
        home = self.workspace.home_path
        run = contained(home, run, directory=True)
        relative = run.relative_to(home / "evaluations")
        if len(relative.parts) != 3:
            raise ValueError("Expected one evaluation result directory")
        return run

    def _index(self, *, rebuild: bool = False) -> HistoryIndex:
        home = self.workspace.home_path
        path = contained(home, "evaluations/history.json")
        if path.exists() and not rebuild:
            return HistoryIndex.model_validate(read_json(path))
        records = []
        for manifest in contained(home, "evaluations", directory=True).glob("*/*/*/manifest.json"):
            record = result_record(home, contained(home, manifest.parent, directory=True))
            if record is not None:
                records.append(record)
        return HistoryIndex(results=sorted(records, key=lambda record: (record.round, record.run_key)))

    def sampling_state(self) -> dict:
        """Return owned, reconstructed-on-demand observations for the next draw."""
        with self._history_lock():
            state = read_state(self.workspace.home_path)
            return {"round": state["successful_round"], "cases": state["cases"], "queries": state["queries"]}

    def record_started(self, run: Path, policy: SamplingPolicy, plan: dict | None = None) -> dict:
        """Record a started round, independently of whether it later enters successful history."""
        with self._history_lock():
            return start_validation(self.workspace.home_path, self._run_path(run), policy, plan)

    def record(self, run: Path) -> EvaluationRecord | None:
        """Register a successful result once; unsuccessful runs never train the sampler."""
        with self._history_lock():
            home = self.workspace.home_path
            run = self._run_path(run)
            record = result_record(home, run)
            if record is None:
                return None
            index = self._index()
            by_key = {item.run_key: item for item in index.results}
            by_key[record.run_key] = record
            update_sampling_state(home, run)
            index = HistoryIndex(results=sorted(by_key.values(), key=lambda item: (item.round, item.run_key)))
            write_json(contained(home, "evaluations/history.json"), index.model_dump(mode="json"))
            return record

    def list_results(self) -> list[EvaluationRecord]:
        """Return successful results in chronological order, including low-scoring candidates."""
        with self._history_lock():
            return self._index().results

    def get_result(self, run_key: str) -> EvaluationRecord:
        """Look up a successful result by its stable workspace-relative identity."""
        for result in self.list_results():
            if result.run_key == run_key:
                return result
        raise KeyError(f"No successful evaluation: {run_key}")

    def rebuild(self) -> list[EvaluationRecord]:
        """Rebuild only derived indexes and sampling observations; never rewrite source evidence."""
        with self._history_lock():
            home = self.workspace.home_path
            index = self._index(rebuild=True)
            rebuild_sampling_state(home)
            write_json(contained(home, "evaluations/history.json"), index.model_dump(mode="json"))
            return index.results

    def invalidate(self) -> None:
        """Discard only derived caches after an unsuccessful registration; retain all source evidence."""
        with self._history_lock():
            for name in ("evaluations/history.json", "sampling/state.json"):
                contained(self.workspace.home_path, name).unlink(missing_ok=True)
