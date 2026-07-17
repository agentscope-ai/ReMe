"""FastMCP STDIO bridge that exposes selected ReMe jobs to Codex."""

import argparse
from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import FunctionTool

from ...config import resolve_app_config
from ...reme import ReMe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose selected ReMe jobs over FastMCP STDIO")
    parser.add_argument("--config", default="default", help="ReMe config name or file path")
    parser.add_argument("--workspace", required=True, help="ReMe workspace directory")
    parser.add_argument("--jobs", required=True, help="JSON array of ReMe job names")
    parser.add_argument("--tool-context-id", default="", help="Context id injected into every job call")
    return parser.parse_args()


def _load_job_names(raw: str) -> list[str]:
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(name, str) and name for name in value):
        raise ValueError("--jobs must be a JSON array of non-empty strings")
    return value


def _make_tool(job: Any, tool_context_id: str) -> FunctionTool:
    async def execute_tool(**kwargs):
        if tool_context_id:
            if "tool_context_id" in kwargs:
                raise ToolError("tool_context_id is managed by the Codex agent wrapper")
            kwargs["tool_context_id"] = tool_context_id
        response = await job(**kwargs)
        if not response.success:
            raise ToolError(str(response.answer))
        return response.answer

    return FunctionTool(
        name=job.name,
        description=job.description,
        fn=execute_tool,
        parameters=job.parameters or {},
    )


def build_server(app: ReMe, job_names: list[str], tool_context_id: str = "") -> FastMCP:
    """Build a STDIO server backed by a dedicated ReMe Application."""

    @asynccontextmanager
    async def lifespan(_server):
        await app.start()
        try:
            yield
        finally:
            await app.close()

    server = FastMCP(name="reme-codex-tools", lifespan=lifespan)
    for name in job_names:
        job = app.context.jobs.get(name)
        if job is None:
            raise KeyError(f"Job '{name}' not found")
        server.add_tool(_make_tool(job, tool_context_id))
    return server


def main() -> None:
    """Load ReMe and serve the requested jobs over STDIO."""
    args = _parse_args()
    job_names = _load_job_names(args.jobs)
    config = resolve_app_config(
        config=args.config,
        workspace_dir=str(Path(args.workspace).absolute()),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        log_config=False,
    )
    # The bridge needs ordinary jobs available for nested job references, but
    # must not start workspace watchers or cron loops in this short-lived child.
    config["jobs"] = {
        name: job_config
        for name, job_config in (config.get("jobs") or {}).items()
        if job_config.get("backend") not in {"background", "cron"}
    }
    app = ReMe(**config)
    server = build_server(app, job_names, args.tool_context_id)
    server.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
