"""Expose ReMe jobs to Codex over FastMCP STDIO.

Multi-workspace callers forward to the resident MCP service without creating
an Application. Single-workspace callers retain the legacy child-service path.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from .base_agent_wrapper import BaseAgentWrapper
from ..client import MCPClient
from ...config import resolve_app_config
from ...reme import ReMe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose selected ReMe jobs over FastMCP STDIO")
    parser.add_argument("--config", default="default", help="ReMe config name or file path")
    parser.add_argument("--workspace", help="Single-workspace directory for the legacy child service")
    parser.add_argument("--workspace-alias", help="Configured alias retained by the legacy child service")
    parser.add_argument("--endpoint", help="Resident streamable-HTTP MCP endpoint for forwarding")
    parser.add_argument("--job", dest="jobs", action="append", required=True, help="ReMe job name; repeat as needed")
    parser.add_argument("--tool-context-id", default="", help="Context id injected into every job call")
    parser.add_argument(
        "--injected-job-kwargs",
        default="",
        help="JSON object of server-owned kwargs injected into every job call",
    )
    parser.add_argument("--tool-timeout", type=float, default=300, help="Per-call timeout in seconds")
    return parser.parse_args()


def _prepare_config(
    config: dict[str, Any],
    job_names: list[str],
    tool_context_id: str = "",
    injected_job_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Configure the legacy single-workspace child Application for MCP STDIO."""
    selected = set(job_names)
    jobs: dict[str, dict[str, Any]] = {}
    for name, raw_job_config in (config.get("jobs") or {}).items():
        job_config = dict(raw_job_config)
        if job_config.get("backend") in {"background", "cron"}:
            continue
        if name in selected:
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
    injected = dict(injected_job_kwargs or {})
    if tool_context_id:
        injected["tool_context_id"] = tool_context_id
    if injected:
        service["injected_job_kwargs"] = injected

    prepared = dict(config)
    prepared["jobs"] = jobs
    prepared["service"] = service
    return prepared


class _ResidentMCPClient:
    """Reuse the standard ReMe MCP client for resident tool discovery and calls."""

    def __init__(self, endpoint: str, timeout: float) -> None:
        from fastmcp.client import StreamableHttpTransport

        self._client = MCPClient(transport=StreamableHttpTransport(url=endpoint), timeout=timeout)

    async def start(self) -> None:
        """Connect to the resident service."""
        await self._client.start()

    async def list_jobs(self) -> list[dict[str, Any]]:
        """Return the resident service's tool descriptors."""
        return await self._client.list_actions()

    async def call(self, job: str, kwargs: dict[str, Any], injected: dict[str, Any]) -> str:
        """Call one resident tool with server-owned arguments enforced."""
        # pylint: disable=protected-access
        payload = BaseAgentWrapper._merge_injected_job_kwargs(kwargs, injected)
        async for text in self._client(action=job, **payload):
            return text
        return ""

    async def close(self) -> None:
        """Close the resident connection."""
        await self._client.close()


def _injected_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    injected = json.loads(args.injected_job_kwargs) if args.injected_job_kwargs else {}
    if not isinstance(injected, dict):
        raise TypeError("--injected-job-kwargs must be a JSON object")
    injected = dict(injected)
    if args.tool_context_id:
        injected["tool_context_id"] = args.tool_context_id
    return injected


async def _run_forwarder(args: argparse.Namespace, injected: dict[str, Any]) -> None:
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from fastmcp.tools import FunctionTool

    client = _ResidentMCPClient(args.endpoint, args.tool_timeout)
    await client.start()
    try:
        metadata = {entry["name"]: entry for entry in await client.list_jobs()}
        missing = sorted(set(args.jobs) - set(metadata))
        if missing:
            raise KeyError(f"Codex job tools not found on the resident MCP service: {', '.join(missing)}")

        server = FastMCP("reme-codex-bridge")
        for name in args.jobs:
            entry = metadata[name]

            def make_forwarder(job_name: str):
                async def forward(**kwargs):
                    try:
                        return await client.call(job_name, kwargs, injected)
                    except Exception as exc:  # noqa: BLE001 - surface every failure as a tool error
                        raise ToolError(str(exc)) from exc

                return forward

            parameters = entry.get("inputSchema") or entry.get("parameters")
            # pylint: disable=protected-access
            parameters = BaseAgentWrapper._strip_injected_parameters(parameters, injected)
            server.add_tool(
                FunctionTool(
                    name=name,
                    description=entry.get("description") or "",
                    parameters=parameters or {"type": "object", "properties": {}},
                    fn=make_forwarder(name),
                ),
            )
        await server.run_async(transport="stdio")
    finally:
        await client.close()


def main() -> None:
    """Forward to a resident service or start the legacy single-workspace child service."""
    args = _parse_args()
    injected = _injected_kwargs(args)
    if args.endpoint:
        asyncio.run(_run_forwarder(args, injected))
        return
    if not args.workspace:
        raise ValueError("--workspace is required when --endpoint is not supplied")
    workspace = str(Path(args.workspace).absolute())
    workspace_dir = {args.workspace_alias: workspace} if args.workspace_alias else workspace
    config = resolve_app_config(
        config=args.config,
        workspace_dir=workspace_dir,
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        log_config=False,
    )
    config = _prepare_config(config, args.jobs, args.tool_context_id, injected)
    ReMe(**config).run_app()


if __name__ == "__main__":
    main()
