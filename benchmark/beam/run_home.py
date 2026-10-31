"""Prepare the BEAM Home workspace, or optimize it with --optimize after preparation."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
import yaml

# Support direct execution from any working directory as well as module imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
from reme_home.preparation.spec import PreparationSpec  # noqa: E402
from reme_home.judge import JUDGE_PLUGINS  # noqa: E402
from reme_home.sampling import SamplingPolicy  # noqa: E402

from reme.application import Application  # noqa: E402

# pylint: enable=wrong-import-position

DEFAULT_CONFIG = Path(__file__).with_name("config_home.yaml")


class RunHomeConfig(BaseModel):
    """Runner settings kept separate from Home's public preparation contract."""

    model_config = ConfigDict(extra="forbid", strict=True)
    workspace_dir: str = Field(min_length=1)
    preparation: dict
    sampling_policy: SamplingPolicy = Field(default_factory=SamplingPolicy)
    validation_cycle_time_budget_minutes: float | None = Field(default=150, gt=0, allow_inf_nan=False)
    max_validations: int = Field(default=10, ge=1)


def project_path(value: str, project_root: Path) -> Path:
    """Use repository-relative paths consistently regardless of the caller's cwd."""
    if value.startswith("~"):
        raise ValueError("Use absolute or project-relative paths, not home-relative paths")
    return (project_root / value).resolve()


def _load_config(config_path: str | Path, *, workspace_dir: str | None = None, project_root: Path = PROJECT_ROOT):
    """Resolve source paths and validate explicit Home preparation parameters."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = RunHomeConfig.model_validate(raw)
    config.preparation.setdefault("candidate", {"paths": ["plugins/beam/src"]})
    spec = PreparationSpec.model_validate(config.preparation)
    spec.benchmark.source = str(project_path(spec.benchmark.source, project_root))
    if spec.benchmark.evidence_source is not None:
        spec.benchmark.evidence_source = str(project_path(spec.benchmark.evidence_source, project_root))
    spec.runtime.source_repo = str(project_path(spec.runtime.source_repo, project_root))
    workspace = project_path(workspace_dir if workspace_dir is not None else config.workspace_dir, project_root)
    return workspace, spec.model_dump(mode="json"), config


def load_config(config_path: str | Path, *, workspace_dir: str | None = None, project_root: Path = PROJECT_ROOT):
    """Return resolved preparation inputs, preserving the preparation-only caller contract."""
    workspace, spec, _ = _load_config(config_path, workspace_dir=workspace_dir, project_root=project_root)
    return workspace, spec


async def _run(config_path, *, workspace_dir=None, optimize_workspace=False, instruction=None):
    """Apply the same runner configuration to each preparation or optimization lifecycle."""
    workspace, spec, config = _load_config(config_path, workspace_dir=workspace_dir)
    app = Application(
        workspace_dir=str(workspace),
        plugins=["home", JUDGE_PLUGINS[spec["benchmark"]["name"]]],
        components={
            "home.evaluation": {"default": {"sampling_policy": config.sampling_policy.model_dump(mode="json")}},
            "agent_wrapper": {
                "home_optimizer": {
                    "validation_cycle_time_budget_minutes": config.validation_cycle_time_budget_minutes,
                    "max_validations": config.max_validations,
                },
            },
        },
        service={"backend": "cli"},
        enable_logo=False,
        log_to_file=False,
    )
    try:
        await app.start()
        if optimize_workspace:
            arguments = {} if instruction is None else {"instruction": instruction}
            response = await app.run_job("home_optimize", **arguments)
        else:
            response = await app.run_job("home_prepare", spec=spec)
        if not response.success:
            raise RuntimeError(response.answer)
        return {"workspace_dir": str(workspace), **response.answer}
    finally:
        await app.close()


async def prepare(config_path: str | Path, *, workspace_dir: str | None = None) -> dict:
    """Prepare once without running evaluations or optimization."""
    return await _run(config_path, workspace_dir=workspace_dir)


async def optimize(config_path: str | Path = DEFAULT_CONFIG, *, workspace_dir=None, instruction=None) -> dict:
    """Optimize an already prepared workspace using this config's actual model budgets."""
    return await _run(config_path, workspace_dir=workspace_dir, optimize_workspace=True, instruction=instruction)


async def run(config_path: str | Path = DEFAULT_CONFIG, *, workspace_dir: str | None = None) -> dict:
    """Prepare through the same Home Job as the CLI, then close the application."""
    return await prepare(config_path, workspace_dir=workspace_dir)


def main(default_config: Path = DEFAULT_CONFIG) -> None:
    """Run preparation or explicit local one-shot optimization with the benchmark config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config, help="Home runner YAML configuration")
    parser.add_argument("--workspace-dir", help="Override workspace directory (absolute or project-relative)")
    parser.add_argument("--optimize", action="store_true", help="Optimize an already prepared workspace")
    parser.add_argument("--instruction", help="Optimization focus; requires --optimize")
    args = parser.parse_args()
    if args.instruction is not None and not args.optimize:
        parser.error("--instruction requires --optimize")
    if args.optimize:
        result = asyncio.run(optimize(args.config, workspace_dir=args.workspace_dir, instruction=args.instruction))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    result = asyncio.run(prepare(args.config, workspace_dir=args.workspace_dir))
    print(
        json.dumps(
            {
                "workspace_dir": result["workspace_dir"],
                "case_count": result["dataset"]["case_count"],
                "query_count": result["dataset"]["query_count"],
                "source_commit": result["code"]["source_commit"],
                "initial_commit": result["code"]["initial_commit"],
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
