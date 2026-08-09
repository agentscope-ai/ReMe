"""Prepare a Meta-ReMe workspace and run its initial validation.

This is the end-to-end Meta-ReMe entry point: it reads the benchmark YAML,
installs the selected dataset cases, builds the initial candidate repository,
and then calls :func:`validation.run_validation` for the ``init`` branch.

It is distinct from ``meta-reme/validation/run.py``, whose CLI only validates
an already prepared workspace and does not install data or create candidate
code. The validation scheduling implementation itself lives in
``validation/evaluator.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable

import yaml

from bundle_builder import build_bundle
from data_preparation import prepare_dataset
from git_manager import initialize_repository
from models import (
    BudgetSpec,
    DatasetSpec,
    DomainSpec,
    ProposerSpec,
    SandboxSpec,
    ScopeSpec,
)
from validation import run_validation
from workspace import Workspace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = {
    "beam": PROJECT_ROOT / "benchmark/beam/dataset/BEAM",
    "longmemeval": PROJECT_ROOT / "benchmark/longmemeval/dataset/longmemeval_s_reme_cleaned.json",
}
INITIAL_CODE_DIR = Path("code/repo/reme")
INITIAL_CODE_ID = "init"
INITIAL_VALIDATION_ID = "initial"
ValidationRunner = Callable[..., Path]


def _domain_spec(dataset: str, source: Path, source_fingerprint: str) -> DomainSpec:
    return DomainSpec(
        dataset=DatasetSpec(name=dataset, source=str(source.resolve()), fingerprint=source_fingerprint),
        bundle_target="beam" if dataset == "beam" else "lme",
        benchmark_runner=f"benchmark.{'beam' if dataset == 'beam' else 'longmemeval'}.run",
        scorer="mean_query_score",
        scope=ScopeSpec(),
        sandbox=SandboxSpec(image="reme-sandbox-base:agentscope-2.0.4-post1", timeout_seconds=3600),
        proposer=ProposerSpec(model="not-configured"),
        budget=BudgetSpec(max_proposals=1),
    )


def prepare_workspace(
    meta_workspace: Path,
    dataset: str,
    train_case_ids: Iterable[str] | None = None,
    dataset_variant: str | None = None,
    dataset_source: Path | None = None,
) -> Workspace:
    """Create/open a workspace and install its training data and initial code bundle."""

    meta_workspace = Path(meta_workspace).resolve()
    source = Path(dataset_source or DEFAULT_SOURCES.get(dataset, "")).resolve()
    with tempfile.TemporaryDirectory(prefix="meta-reme-dataset-") as temporary:
        normalized = Path(temporary)
        manifest = prepare_dataset(
            normalized,
            dataset,
            source,
            train_case_ids=train_case_ids,
            variant=dataset_variant,
        )
        spec = _domain_spec(dataset, source, manifest.source_fingerprint)
        is_empty_directory = meta_workspace.is_dir() and not any(meta_workspace.iterdir())
        workspace = (
            Workspace.create(meta_workspace, spec)
            if not meta_workspace.exists() or is_empty_directory
            else Workspace.open(meta_workspace, spec)
        )
        with workspace.acquire_lock():
            if not workspace.path("dataset/manifest.json").exists():
                workspace.install_dataset(normalized)
            _prepare_initial_code(workspace, spec.bundle_target)
    return workspace


def _prepare_initial_code(workspace: Workspace, bundle_target: str) -> Path:
    """Build the benchmark bundle once without overwriting an existing repository."""

    bundle = workspace.path(INITIAL_CODE_DIR)
    repository_parent = bundle.parent
    if bundle.is_dir() and (bundle / "pyproject.toml").is_file():
        initialize_repository(bundle)
        return bundle
    if any(repository_parent.iterdir()):
        raise RuntimeError(f"Initial code directory is not empty or complete: {repository_parent}")
    bundle = build_bundle(bundle_target, repository_parent, source_repo=PROJECT_ROOT)
    initialize_repository(bundle)
    return bundle


def run_initial_validation(
    workspace: Workspace,
    concurrency: int,
    *,
    fail_fast: bool = False,
    validation_runner: ValidationRunner = run_validation,
) -> Path:
    """Validate the initial code branch once against every installed case."""

    if concurrency < 1:
        raise ValueError("validation.concurrency must be at least 1")
    output = workspace.path(f"evaluations/{INITIAL_CODE_ID}/{INITIAL_VALIDATION_ID}")
    if (output / "summary.json").is_file():
        return output
    case_ids = []
    for case_path in sorted(workspace.path("dataset/cases").glob("*.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Prepared case has no valid case_id: {case_path}")
        case_ids.append(case_id)
    if not case_ids:
        raise ValueError("Prepared dataset has no cases to validate")
    return validation_runner(
        workspace.root,
        case_ids,
        INITIAL_CODE_ID,
        concurrency,
        validation_id=INITIAL_VALIDATION_ID,
        fail_fast=fail_fast,
    )


def prepare_and_validate_workspace(
    config_path: Path,
    *,
    validation_runner: ValidationRunner = run_validation,
) -> tuple[Workspace, Path]:
    """Prepare the configured workspace and run its first full validation."""

    config = load_config(config_path)
    workspace = prepare_workspace(
        meta_workspace=config["meta_workspace"],
        dataset=config["dataset"],
        train_case_ids=config["train_case_ids"],
        dataset_variant=config["dataset_variant"],
        dataset_source=config["dataset_source"],
    )
    validation = run_initial_validation(
        workspace,
        config["validation_concurrency"],
        fail_fast=config["validation_fail_fast"],
        validation_runner=validation_runner,
    )
    return workspace, validation


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return value


def _project_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the dataset-preparation portion of a Meta-ReMe config."""

    raw = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    dataset_config = _mapping(raw.get("dataset"), "dataset")
    dataset = dataset_config.get("name")
    if dataset not in DEFAULT_SOURCES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    case_ids = dataset_config.get("train_case_ids", [])
    if not isinstance(case_ids, list) or any(not isinstance(case_id, (str, int)) for case_id in case_ids):
        raise ValueError("dataset.train_case_ids must be a list of strings or integers")
    source_value = dataset_config.get("source")
    source = DEFAULT_SOURCES[dataset] if source_value is None else _project_path(source_value, "dataset.source")
    validation_config = raw.get("validation", {})
    validation_config = _mapping(validation_config, "validation")
    validation_concurrency = validation_config.get("concurrency", 1)
    if (
        not isinstance(validation_concurrency, int)
        or isinstance(validation_concurrency, bool)
        or validation_concurrency < 1
    ):
        raise ValueError("validation.concurrency must be a positive integer")
    validation_fail_fast = validation_config.get("fail_fast", False)
    if not isinstance(validation_fail_fast, bool):
        raise ValueError("validation.fail_fast must be a boolean")
    return {
        "meta_workspace": _project_path(raw.get("meta_workspace"), "meta_workspace"),
        "dataset": dataset,
        "train_case_ids": [str(case_id) for case_id in case_ids],
        "dataset_variant": dataset_config.get("variant"),
        "dataset_source": source,
        "validation_concurrency": validation_concurrency,
        "validation_fail_fast": validation_fail_fast,
    }


def parse_args() -> argparse.Namespace:
    """Parse the single configuration-file argument."""

    parser = argparse.ArgumentParser(description="Initialize Meta-ReMe and normalize its training set")
    parser.add_argument("--config", type=Path, required=True, help="Path to config_meta_reme.yaml")
    return parser.parse_args()


def main() -> None:
    """Prepare the configured workspace and validate all data with its initial code."""

    args = parse_args()
    workspace, validation = prepare_and_validate_workspace(args.config)
    manifest = json.loads(workspace.path("dataset/manifest.json").read_text(encoding="utf-8"))
    print(
        f"Meta-ReMe workspace ready: {workspace.root} "
        f"({manifest['case_count']} training cases, {manifest['query_count']} queries); "
        f"initial validation: {validation}",
    )


if __name__ == "__main__":
    main()
