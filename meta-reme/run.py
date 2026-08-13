"""Prepare a Meta-ReMe workspace and run its initial validation.

This is the end-to-end Meta-ReMe entry point: it reads the benchmark YAML,
installs the selected dataset cases, builds the initial candidate repository,
and then calls :func:`validation.run_validation` for the current clean commit.

It is distinct from ``meta-reme/validation/run.py``, whose CLI only validates
an already prepared workspace and does not install data or create candidate
code. The validation scheduling implementation itself lives in
``validation/evaluator.py``.
"""

from __future__ import annotations

import argparse
import asyncio
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterable, NamedTuple

import yaml
from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from runtime import TOOL_RUNTIME
from validation import resolve_current_revision, run_validation
from workspace import Workspace

DEFAULT_SOURCES = {
    "beam": PROJECT_ROOT / "benchmark/beam/dataset/BEAM",
    "longmemeval": PROJECT_ROOT / "benchmark/longmemeval/dataset/longmemeval_s_reme_cleaned.json",
}
INITIAL_CODE_DIR = Path("code/repo/reme")
INITIAL_VALIDATION_ID = "initial"
ValidationRunner = Callable[..., Path]
OptimizerFactory = Callable[..., Any]
OptimizerRunner = Callable[..., Any]


class SearchConfig(NamedTuple):
    """Validated configuration for one optimizer-agent search."""

    enabled: bool = True
    max_agent_iters: int = 80
    max_code_iterations: int = 5
    objective: str = "Improve mean_query_score on every installed search case."


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
    """Validate the initial code commit once against every installed case."""

    if concurrency < 1:
        raise ValueError("validation.concurrency must be at least 1")
    branch_name, commit_sha = resolve_current_revision(workspace.path(INITIAL_CODE_DIR))
    output = workspace.validation_commit_dir(branch_name, commit_sha) / INITIAL_VALIDATION_ID
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
    TOOL_RUNTIME.configure(workspace.root, config["validation_concurrency"])
    validation = run_initial_validation(
        workspace,
        config["validation_concurrency"],
        fail_fast=config["validation_fail_fast"],
        validation_runner=validation_runner,
    )
    return workspace, validation


def require_complete_search_validation(workspace: Workspace, validation: Path) -> dict[str, Any]:
    """Reject an incomplete baseline before any optimizer is allowed to edit code."""

    summary_path = validation / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Initial validation did not publish a summary: {summary_path}")
    summary = _mapping(json.loads(summary_path.read_text(encoding="utf-8")), "initial validation summary")
    installed_case_ids = []
    for case_path in sorted(workspace.path("dataset/cases").glob("*.json")):
        case = _mapping(json.loads(case_path.read_text(encoding="utf-8")), f"prepared case {case_path}")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError(f"Prepared case has no valid case_id: {case_path}")
        installed_case_ids.append(case_id)
    completed_case_ids = [case.get("case_id") for case in summary.get("cases", []) if isinstance(case, dict)]
    if summary.get("status") != "completed" or completed_case_ids != installed_case_ids:
        raise RuntimeError(
            "Initial validation must complete every installed search case before optimization: "
            f"status={summary.get('status')!r}, expected={installed_case_ids!r}, completed={completed_case_ids!r}",
        )
    return summary


def create_search_model(model_name: str) -> OpenAIChatModel:
    """Create one YAML-configured agent model from the OpenAI-compatible LLM environment."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Search requires LLM_API_KEY in the environment or project .env")
    credential = OpenAICredential(api_key=api_key, base_url=os.environ.get("LLM_BASE_URL") or None)
    return OpenAIChatModel(credential=credential, model=model_name, stream=True)


async def optimize_validated_workspace(
    workspace: Workspace,
    validation: Path,
    config: SearchConfig,
    *,
    validation_concurrency: int,
    optimizer_factory: OptimizerFactory | None = None,
    optimizer_runner: OptimizerRunner | None = None,
) -> Any:
    """Run the main optimizer only after the complete initial search validation."""

    require_complete_search_validation(workspace, validation)
    if not config.enabled:
        return None
    optimizer_module = import_module("as.agent.optimizer_agent")
    diagnostic_module = import_module("as.agent.diagnostic_agent")
    factory = optimizer_factory or optimizer_module.create_optimizer_agent
    runner = optimizer_runner or optimizer_module.run_optimizer_agent
    objective = (
        f"{config.objective.rstrip()} Perform at most {config.max_code_iterations} committed code iterations in this "
        "run; count each distinct candidate commit as one iteration and stop after that limit."
    )
    optimizer_model = create_search_model(optimizer_module.load_agent_prompt(optimizer_module.__file__).model)
    diagnostic_model = create_search_model(diagnostic_module.load_agent_prompt(diagnostic_module.__file__).model)
    agent = factory(
        optimizer_model,
        workspace.root,
        diagnostic_model=diagnostic_model,
        validation_concurrency=validation_concurrency,
        max_iters=config.max_agent_iters,
    )
    return await runner(agent, objective=objective)


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
    search_config = _load_search_config(raw.get("search", {}))
    return {
        "meta_workspace": _project_path(raw.get("meta_workspace"), "meta_workspace"),
        "dataset": dataset,
        "train_case_ids": [str(case_id) for case_id in case_ids],
        "dataset_variant": dataset_config.get("variant"),
        "dataset_source": source,
        "validation_concurrency": validation_concurrency,
        "validation_fail_fast": validation_fail_fast,
        "search": search_config,
    }


def _load_search_config(value: Any) -> SearchConfig:
    config = _mapping(value, "search")
    unknown = set(config) - {
        "enabled",
        "max_agent_iters",
        "max_code_iterations",
        "objective",
    }
    if unknown:
        raise ValueError(f"Unknown search configuration fields: {sorted(unknown)}")
    enabled = config.get("enabled", True)
    max_agent_iters = config.get("max_agent_iters", 80)
    max_code_iterations = config.get("max_code_iterations", 5)
    objective = config.get("objective", "Improve mean_query_score on every installed search case.")
    if not isinstance(enabled, bool):
        raise ValueError("search.enabled must be a boolean")
    for field, number in (("max_agent_iters", max_agent_iters), ("max_code_iterations", max_code_iterations)):
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError(f"search.{field} must be a positive integer")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("search.objective must be a non-empty string")
    return SearchConfig(
        enabled,
        max_agent_iters,
        max_code_iterations,
        objective.strip(),
    )


def parse_args() -> argparse.Namespace:
    """Parse the single configuration-file argument."""

    parser = argparse.ArgumentParser(description="Initialize Meta-ReMe and normalize its training set")
    parser.add_argument("--config", type=Path, required=True, help="Path to config_meta_reme.yaml")
    return parser.parse_args()


def run(config_path: Path) -> tuple[Workspace, Path, Any]:
    """Prepare, fully validate, then optimize the configured search set."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = load_config(config_path)
    workspace, validation = prepare_and_validate_workspace(config_path)
    result = asyncio.run(
        optimize_validated_workspace(
            workspace,
            validation,
            config["search"],
            validation_concurrency=config["validation_concurrency"],
        ),
    )
    return workspace, validation, result


def main() -> None:
    """Prepare, validate all search cases, and invoke the main optimizer agent."""

    args = parse_args()
    workspace, validation, result = run(args.config)
    manifest = json.loads(workspace.path("dataset/manifest.json").read_text(encoding="utf-8"))
    print(
        f"Meta-ReMe workspace ready: {workspace.root} "
        f"({manifest['case_count']} training cases, {manifest['query_count']} queries); "
        f"initial validation: {validation}",
    )
    if result is not None:
        print(result.get_text_content() or result)


if __name__ == "__main__":
    main()
