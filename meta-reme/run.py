"""Initialize a Meta-ReMe workspace from a benchmark YAML configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import yaml

from data_preparation import prepare_dataset
from models import (
    BudgetSpec,
    DatasetSpec,
    DomainSpec,
    ProposerSpec,
    SandboxSpec,
    ScopeSpec,
)
from workspace import Workspace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = {
    "beam": PROJECT_ROOT / "benchmark/beam/dataset/BEAM",
    "longmemeval": PROJECT_ROOT / "benchmark/longmemeval/dataset/longmemeval_s_reme_cleaned.json",
}


def _domain_spec(dataset: str, source: Path, source_fingerprint: str) -> DomainSpec:
    return DomainSpec(
        dataset=DatasetSpec(name=dataset, source=str(source.resolve()), fingerprint=source_fingerprint),
        bundle_target="beam" if dataset == "beam" else "lme",
        benchmark_runner=f"benchmark.{'beam' if dataset == 'beam' else 'longmemeval'}.run",
        scorer="mean_query_score",
        scope=ScopeSpec(),
        sandbox=SandboxSpec(image="reme:latest", timeout_seconds=3600),
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
    """Create/open a workspace and install only the selected training cases."""

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
            if not workspace.path("datasets/search/manifest.json").exists():
                workspace.install_search_dataset(normalized)
    return workspace


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
    return {
        "meta_workspace": _project_path(raw.get("meta_workspace"), "meta_workspace"),
        "dataset": dataset,
        "train_case_ids": [str(case_id) for case_id in case_ids],
        "dataset_variant": dataset_config.get("variant"),
        "dataset_source": source,
    }


def parse_args() -> argparse.Namespace:
    """Parse the single configuration-file argument."""

    parser = argparse.ArgumentParser(description="Initialize Meta-ReMe and normalize its training set")
    parser.add_argument("--config", type=Path, required=True, help="Path to config_meta_reme.yaml")
    return parser.parse_args()


def main() -> None:
    """Prepare the configured workspace and print a short dataset summary."""

    args = parse_args()
    workspace = prepare_workspace(**load_config(args.config))
    manifest = json.loads(workspace.path("datasets/search/manifest.json").read_text(encoding="utf-8"))
    print(
        f"Meta-ReMe workspace ready: {workspace.root} "
        f"({manifest['case_count']} training cases, {manifest['query_count']} queries)",
    )


if __name__ == "__main__":
    main()
