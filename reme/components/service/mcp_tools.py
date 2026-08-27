"""Shared MCP tool registration for services that expose ReMe jobs."""

from typing import Any

from ..job import BaseJob, StreamJob


def add_mcp_job(
    server: Any,
    job: BaseJob,
    *,
    injected_job_kwargs: dict[str, Any],
    tool_error_on_failure: bool,
) -> bool:
    """Register a non-stream job as an MCP tool on ``server``."""
    from fastmcp.exceptions import ToolError
    from fastmcp.tools import FunctionTool

    if isinstance(job, StreamJob):
        return False

    async def execute_tool(**kwargs):
        conflicts = sorted(injected_job_kwargs.keys() & kwargs.keys())
        if conflicts:
            names = ", ".join(conflicts)
            raise ToolError(
                f"{names} injected by the MCP server and cannot be provided by the caller",
            )
        kwargs.update(injected_job_kwargs)
        response = await job(**kwargs)
        if tool_error_on_failure and not response.success:
            raise ToolError(str(response.answer))
        return response.answer

    parameters = dict(job.parameters or {})
    injected_names = injected_job_kwargs.keys()
    if "properties" in parameters:
        parameters["properties"] = {
            name: schema for name, schema in parameters["properties"].items() if name not in injected_names
        }
    if "required" in parameters:
        parameters["required"] = [name for name in parameters["required"] if name not in injected_names]

    server.add_tool(
        FunctionTool(
            name=job.name,
            description=job.description,
            fn=execute_tool,
            parameters=parameters,
        ),
    )
    return True
