"""ReMe MCP calls shared by hooks and the plugin's connection check."""

from __future__ import annotations

import asyncio


async def call_async(config: dict, action: str, payload: dict, timeout: float) -> dict:
    """Acknowledge a write only after a successful MCP result, including connection time."""
    from fastmcp import Client

    async with asyncio.timeout(timeout):
        async with Client(config["mcp_url"], timeout=timeout) as client:
            result = await client.call_tool(action, payload, raise_on_error=True)
    if result.is_error:
        raise RuntimeError("ReMe MCP did not acknowledge the action")
    answer = result.data
    if answer is None:
        answer = "\n".join(block.text for block in result.content if block.type == "text")
    return {"success": True, "answer": answer}


def call(config: dict, action: str, payload: dict, timeout: float) -> dict:
    """Run one bounded MCP operation in a short-lived native hook process."""
    try:
        return asyncio.run(call_async(config, action, payload, timeout))
    except Exception as exc:
        # Keep the hook's retry loop independent of MCP transport exception classes.
        raise RuntimeError("ReMe MCP request failed") from exc
