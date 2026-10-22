"""Thin FastMCP STDIO bridge forwarding Codex job tools to the parent process.

This subprocess never builds a ReMe Application or any workspace store. It
asks the parent-owned loopback endpoint (LoopbackJobServer) for job metadata
and forwards every tool call — together with the server-owned workspace alias,
tool context id, and injected kwargs — back to the parent for execution, so
all workspace state has exactly one owner.
"""

import argparse
import json
from typing import Any

import httpx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward selected ReMe jobs from the parent loopback endpoint")
    parser.add_argument("--endpoint", required=True, help="Parent loopback endpoint base URL")
    parser.add_argument("--token", required=True, help="Bearer token for the loopback endpoint")
    parser.add_argument("--job", dest="jobs", action="append", required=True, help="ReMe job name; repeat as needed")
    parser.add_argument("--workspace-alias", default="", help="Workspace alias injected into every job call")
    parser.add_argument("--tool-context-id", default="", help="Context id injected into every job call")
    parser.add_argument(
        "--injected-job-kwargs",
        default="",
        help="JSON object of server-owned kwargs injected into every job call",
    )
    parser.add_argument("--tool-timeout", type=float, default=300, help="Per-call timeout in seconds")
    return parser.parse_args()


class _EndpointClient:
    """Minimal client for the parent's loopback job server."""

    def __init__(self, endpoint: str, token: str, timeout: float) -> None:
        self._client = httpx.Client(
            base_url=endpoint,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        """Fetch the servable job metadata from the parent endpoint."""
        response = self._client.get("/jobs")
        response.raise_for_status()
        return response.json()["jobs"]

    def call(self, job: str, kwargs: dict[str, Any], injected_job_kwargs: dict[str, Any] | None = None) -> str:
        """Forward one tool call to the parent endpoint and unwrap its answer."""
        response = self._client.post(
            "/call",
            json={"job": job, "kwargs": kwargs, "injected_job_kwargs": injected_job_kwargs or {}},
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("error") or response.text
            except ValueError:
                detail = response.text
            raise RuntimeError(str(detail))
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(str(payload.get("answer") or payload))
        return str(payload.get("answer") or "")

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


def main() -> None:
    """Fetch job metadata from the parent and serve forwarding tools over STDIO."""
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from fastmcp.tools import FunctionTool

    args = _parse_args()
    injected_job_kwargs = json.loads(args.injected_job_kwargs) if args.injected_job_kwargs else {}
    if not isinstance(injected_job_kwargs, dict):
        raise TypeError("--injected-job-kwargs must be a JSON object")
    injected = dict(injected_job_kwargs)
    if args.workspace_alias:
        injected["workspace"] = args.workspace_alias
    if args.tool_context_id:
        injected["tool_context_id"] = args.tool_context_id

    client = _EndpointClient(args.endpoint, args.token, args.tool_timeout)
    metadata = {entry["name"]: entry for entry in client.list_jobs()}
    missing = sorted(set(args.jobs) - set(metadata))
    if missing:
        raise KeyError(f"Codex job tools not found on the parent endpoint: {', '.join(missing)}")

    server = FastMCP("reme-codex-bridge")
    for name in args.jobs:
        entry = metadata[name]

        def forward(_client=client, _name=name, _injected=None, **kwargs):
            try:
                return _client.call(_name, kwargs, _injected or injected)
            except Exception as exc:  # noqa: BLE001 - surface every failure as a tool error
                raise ToolError(str(exc)) from exc

        server.add_tool(
            FunctionTool(
                name=name,
                description=entry.get("description") or "",
                parameters=entry.get("parameters") or {"type": "object", "properties": {}},
                fn=forward,
            ),
        )

    try:
        server.run(transport="stdio")
    finally:
        client.close()


if __name__ == "__main__":
    main()
