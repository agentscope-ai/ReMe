"""FastMCP STDIO bridge that exposes selected ReMe jobs to Codex."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...config import ResolvedAppConfig, deep_merge_config, resolve_app_config_layers
from ...reme import ReMe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose selected ReMe jobs over FastMCP STDIO")
    parser.add_argument("--config", default="default", help="ReMe config name or file path")
    parser.add_argument("--workspace", required=True, help="ReMe workspace directory")
    parser.add_argument("--job", dest="jobs", action="append", required=True, help="ReMe job name; repeat as needed")
    parser.add_argument("--tool-context-id", default="", help="Context id injected into every job call")
    parser.add_argument(
        "--injected-job-kwargs",
        default="",
        help="JSON object of server-owned kwargs injected into every job call",
    )
    return parser.parse_args()


def _prepare_config(
    config: ResolvedAppConfig | Mapping[str, Any],
    job_names: list[str],
    tool_context_id: str = "",
    injected_job_kwargs: dict[str, Any] | None = None,
) -> ResolvedAppConfig | dict[str, Any]:
    """Configure the dedicated child Application to serve selected jobs over MCP STDIO."""
    return_layers = isinstance(config, ResolvedAppConfig)
    resolved = config if return_layers else ResolvedAppConfig(overrides=config)
    visible = resolved.materialize()
    selected = set(job_names)
    allowed_jobs: set[str] = set()
    for name, raw_job_config in (visible.get("jobs") or {}).items():
        job_config = dict(raw_job_config)
        if job_config.get("backend") in {"background", "cron"}:
            continue
        allowed_jobs.add(name)

    missing = sorted(selected.difference(allowed_jobs))
    if missing:
        raise KeyError(f"Codex job tools not found or not request jobs: {', '.join(missing)}")

    service: dict[str, Any] = {
        "backend": "mcp",
        "transport": "stdio",
        "jobs": job_names,
        "tool_error_on_failure": True,
    }
    injected = dict(injected_job_kwargs or {})
    if tool_context_id:
        injected["tool_context_id"] = tool_context_id
    if injected:
        service["injected_job_kwargs"] = injected

    base = dict(resolved.base)
    if isinstance(base.get("jobs"), Mapping):
        base["jobs"] = {name: value for name, value in base["jobs"].items() if name in allowed_jobs}

    overrides = dict(resolved.overrides)
    if isinstance(overrides.get("jobs"), Mapping):
        overrides["jobs"] = {name: value for name, value in overrides["jobs"].items() if name in allowed_jobs}

    # Job selection is an explicit server-owned patch. It must win after a
    # plugin atomically replaces a loaded Job with the same name.
    selected_patch = {name: {"enable_serve": True} for name in job_names}
    overrides = deep_merge_config(overrides, {"jobs": selected_patch, "service": service})
    prepared = ResolvedAppConfig(base=base, overrides=overrides)
    return prepared if return_layers else prepared.materialize()


def main() -> None:
    """Load ReMe and serve the requested jobs over STDIO."""
    args = _parse_args()
    config = resolve_app_config_layers(
        config=args.config,
        workspace_dir=str(Path(args.workspace).absolute()),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        log_config=False,
    )
    injected_job_kwargs = json.loads(args.injected_job_kwargs) if args.injected_job_kwargs else None
    if injected_job_kwargs is not None and not isinstance(injected_job_kwargs, dict):
        raise TypeError("--injected-job-kwargs must be a JSON object")
    config = _prepare_config(config, args.jobs, args.tool_context_id, injected_job_kwargs)
    ReMe(resolved_config=config).run_app()


if __name__ == "__main__":
    main()
