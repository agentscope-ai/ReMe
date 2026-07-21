"""FastMCP STDIO bridge that exposes selected ReMe jobs to Codex."""

import argparse
from pathlib import Path
from typing import Any

from ...config import resolve_app_config
from ...reme import ReMe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose selected ReMe jobs over FastMCP STDIO")
    parser.add_argument("--config", default="default", help="ReMe config name or file path")
    parser.add_argument("--workspace", required=True, help="ReMe workspace directory")
    parser.add_argument("--job", dest="jobs", action="append", required=True, help="ReMe job name; repeat as needed")
    parser.add_argument("--tool-context-id", default="", help="Context id injected into every job call")
    return parser.parse_args()


def _prepare_config(config: dict[str, Any], job_names: list[str], tool_context_id: str = "") -> dict[str, Any]:
    """Configure the dedicated child Application to serve selected jobs over MCP STDIO."""
    selected = set(job_names)
    jobs: dict[str, dict[str, Any]] = {}
    for name, raw_job_config in (config.get("jobs") or {}).items():
        job_config = dict(raw_job_config)
        if job_config.get("backend") in {"background", "cron"}:
            continue
        if name in selected:
            # An explicit Codex job_tools selection has always overridden enable_serve.
            job_config["enable_serve"] = True
        jobs[name] = job_config

    missing = sorted(selected.difference(jobs))
    if missing:
        raise KeyError(f"Codex job tools not found or not request jobs: {', '.join(missing)}")

    service: dict[str, Any] = {
        "backend": "mcp",
        "transport": "stdio",
        "jobs": job_names,
        "tool_error_on_failure": True,
    }
    if tool_context_id:
        service["injected_job_kwargs"] = {"tool_context_id": tool_context_id}

    prepared = dict(config)
    prepared["jobs"] = jobs
    prepared["service"] = service
    return prepared


def main() -> None:
    """Load ReMe and serve the requested jobs over STDIO."""
    args = _parse_args()
    config = resolve_app_config(
        config=args.config,
        workspace_dir=str(Path(args.workspace).absolute()),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        log_config=False,
    )
    config = _prepare_config(config, args.jobs, args.tool_context_id)
    ReMe(**config).run_app()


if __name__ == "__main__":
    main()
