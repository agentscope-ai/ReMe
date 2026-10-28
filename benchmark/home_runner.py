"""Shared preparation-only entry point for benchmark Home experiments."""

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
import yaml
from reme_home.preparation.spec import PreparationSpec
from reme_home.judge import JUDGE_PLUGINS

from reme.application import Application

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RunHomeConfig(BaseModel):
    """Runner settings kept separate from Home's public preparation contract."""

    model_config = ConfigDict(extra="forbid", strict=True)
    workspace_dir: str = Field(min_length=1)
    preparation: dict


def project_path(value: str, project_root: Path) -> Path:
    """Use repository-relative paths consistently regardless of the caller's cwd."""
    if value.startswith("~"):
        raise ValueError("Use absolute or project-relative paths, not home-relative paths")
    return (project_root / value).resolve()


def load_config(config_path: str | Path, *, workspace_dir: str | None = None, project_root: Path = PROJECT_ROOT):
    """Resolve source paths and validate explicit Home preparation parameters."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = RunHomeConfig.model_validate(raw)
    spec = PreparationSpec.model_validate(config.preparation)
    spec.benchmark.source = str(project_path(spec.benchmark.source, project_root))
    if spec.benchmark.evidence_source is not None:
        spec.benchmark.evidence_source = str(project_path(spec.benchmark.evidence_source, project_root))
    spec.runtime.source_repo = str(project_path(spec.runtime.source_repo, project_root))
    workspace = project_path(workspace_dir if workspace_dir is not None else config.workspace_dir, project_root)
    return workspace, spec.model_dump(mode="json")


async def prepare(config_path: str | Path, *, workspace_dir: str | None = None) -> dict:
    """Start Home, prepare once, and close all resources without evaluating cases."""
    workspace, spec = load_config(config_path, workspace_dir=workspace_dir)
    app = Application(
        workspace_dir=str(workspace),
        plugins=["home", JUDGE_PLUGINS[spec["benchmark"]["name"]]],
        service={"backend": "cli"},
        enable_logo=False,
        log_to_file=False,
    )
    try:
        await app.start()
        response = await app.run_job("home_prepare", spec=spec)
        if not response.success:
            raise RuntimeError(response.answer)
        return {"workspace_dir": str(workspace), **response.answer}
    finally:
        await app.close()


def main(default_config: Path) -> None:
    """Run a benchmark's default preparation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config, help="Home runner YAML configuration")
    parser.add_argument("--workspace-dir", help="Override workspace directory (absolute or project-relative)")
    args = parser.parse_args()
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
