"""Focused contract and filesystem tests for Meta-ReMe's initial foundation."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import socket
import sys

import pytest
from pydantic import ValidationError

META_REME = Path(__file__).resolve().parents[2] / "meta-reme"
sys.path.insert(0, str(META_REME))
models = importlib.import_module("models")
workspace_module = importlib.import_module("workspace")


def domain_spec():
    return models.DomainSpec(
        dataset=models.DatasetSpec(name="longmemeval", source="dataset.json", fingerprint="dataset-sha"),
        bundle_target="lme",
        benchmark_runner="benchmark.longmemeval.run",
        scorer="lme_answer_judge_step",
        scope=models.ScopeSpec(harness_paths=["reme/steps/benchmark/lme/auto_memory.py"]),
        sandbox=models.SandboxSpec(image="reme:test", timeout_seconds=60),
        proposer=models.ProposerSpec(model="test-model"),
        budget=models.BudgetSpec(max_proposals=2),
    )


def test_fingerprint_is_stable_for_mapping_order() -> None:
    assert models.fingerprint({"a": 1, "b": [2, 3]}) == models.fingerprint({"b": [2, 3], "a": 1})


def test_contracts_reject_unknown_fields_and_invalid_terminal_states() -> None:
    with pytest.raises(ValidationError):
        models.DatasetSpec(name="longmemeval", source="x", fingerprint="sha", typo=True)
    with pytest.raises(ValidationError, match="at least one search budget"):
        models.BudgetSpec()
    with pytest.raises(ValidationError, match="at least one query"):
        models.CaseResult(case_id="case", attempt_id="attempt", status="completed")
    with pytest.raises(ValidationError, match="reusable terminal status"):
        models.AttemptCompletion(
            case_id="case",
            attempt_id="attempt",
            status="infra_error",
            fingerprints=models.Fingerprints(dataset="d", code="c", config="f", model="m", image="i"),
            selection_fingerprint="selection-sha",
            result_sha256="sha",
        )


def test_partial_validation_selection_is_not_comparable() -> None:
    selection = models.ValidationSelection(
        case_ids=["case-1"],
        query_ids={"case-1": ["query-1"]},
        reason="target known failure",
    )
    coverage = models.ValidationCoverage(
        selected_cases=1,
        total_cases=10,
        selected_queries=1,
        total_queries=100,
        is_full=False,
    )
    common = {
        "validation_id": "validation-1",
        "commit_sha": "commit",
        "mode": "screening",
        "status": "completed",
        "fingerprints": models.Fingerprints(dataset="d", code="c", config="f", model="m", image="i"),
        "selection": selection,
        "coverage": coverage,
        "mean_query_score": 1.0,
        "query_count": 1,
    }

    result = models.ValidationResult(**common)
    assert not result.comparable
    spec = models.ValidationSpec(
        commit_sha="commit",
        mode="screening",
        selection=selection,
        fingerprints=common["fingerprints"],
    )
    assert spec.selection_fingerprint == models.model_fingerprint(selection)
    with pytest.raises(ValidationError, match="comparable requires"):
        models.ValidationResult(**common, comparable=True)


def test_validation_selection_rejects_queries_from_unselected_cases() -> None:
    with pytest.raises(ValidationError, match="unselected cases"):
        models.ValidationSelection(case_ids=["case-1"], query_ids={"case-2": ["query-1"]}, reason="targeted")


def test_workspace_create_open_and_domain_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    spec = domain_spec()
    workspace = workspace_module.Workspace.create(root, spec)

    assert workspace.path("code/repo").is_dir()
    assert workspace.path("dataset/cases").is_dir()
    assert not workspace.path("harnesses").exists()
    assert workspace_module.Workspace.open(root, spec).manifest == workspace.manifest

    changed = spec.model_copy(update={"scorer": "different"})
    with pytest.raises(workspace_module.WorkspaceFormatError, match="fingerprint"):
        workspace_module.Workspace.open(root, changed)
    with pytest.raises(workspace_module.WorkspaceError, match="non-empty"):
        workspace_module.Workspace.create(root, spec)


def test_workspace_reports_legacy_dataset_layout(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    workspace = workspace_module.Workspace.create(root, domain_spec())
    workspace.path("datasets").mkdir()
    workspace.path("dataset").rename(workspace.path("datasets/search"))

    with pytest.raises(workspace_module.WorkspaceFormatError, match="move that directory to dataset"):
        workspace_module.Workspace.open(root, domain_spec())


def test_workspace_rejects_escaping_paths_and_ids(tmp_path: Path) -> None:
    workspace = workspace_module.Workspace.create(tmp_path / "workspace", domain_spec())
    with pytest.raises(workspace_module.WorkspaceError):
        workspace.path("../outside")
    with pytest.raises(workspace_module.WorkspaceError):
        workspace.entity_path("evaluations", "../case")

    case_result = workspace.validation_case_dir("init", "validation-1", "case-1", create=True)
    assert case_result == workspace.path("evaluations/init/validation-1/cases/case-1")


def test_atomic_write_replaces_content_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    workspace = workspace_module.Workspace.create(tmp_path / "workspace", domain_spec())
    destination = workspace.atomic_write_json("run.json", {"value": 1})
    workspace.atomic_write_json("run.json", {"value": 2})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"value": 2}
    assert not list(destination.parent.glob(".run.json.*"))


def test_install_dataset_replaces_empty_skeleton_and_is_read_only(tmp_path: Path) -> None:
    workspace = workspace_module.Workspace.create(tmp_path / "workspace", domain_spec())
    source = tmp_path / "normalized"
    (source / "cases").mkdir(parents=True)
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    (source / "cases/case-1.json").write_text("{}\n", encoding="utf-8")

    installed = workspace.install_dataset(source)

    assert (installed / "cases/case-1.json").is_file()
    assert not (installed / "manifest.json").stat().st_mode & 0o222
    with pytest.raises(workspace_module.WorkspaceError, match="already been installed"):
        workspace.install_dataset(source)


def test_workspace_lock_rejects_live_owner_and_releases(tmp_path: Path) -> None:
    workspace = workspace_module.Workspace.create(tmp_path / "workspace", domain_spec())
    lock = workspace.acquire_lock()
    with pytest.raises(workspace_module.WorkspaceLockedError, match="locked by pid"):
        workspace.acquire_lock()
    lock.release()
    with workspace.acquire_lock():
        assert workspace.path(workspace_module.WORKSPACE_LOCK).exists()
    assert not workspace.path(workspace_module.WORKSPACE_LOCK).exists()


def test_workspace_lock_recovers_confirmed_stale_local_owner(tmp_path: Path) -> None:
    workspace = workspace_module.Workspace.create(tmp_path / "workspace", domain_spec())
    stale = models.WorkspaceLockOwner(pid=_missing_pid(), hostname=socket.gethostname())
    workspace.atomic_write_json(workspace_module.WORKSPACE_LOCK, stale)

    with workspace.acquire_lock() as lock:
        assert lock.owner is not None
        assert lock.owner.token != stale.token


def _missing_pid() -> int:
    candidate = max(os.getpid() + 1_000_000, 10_000_000)
    while True:
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        candidate += 1
