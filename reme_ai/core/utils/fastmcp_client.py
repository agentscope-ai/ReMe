import asyncio
import os
import shutil
from typing import List, Optional, Union, Callable, Coroutine, Any

import mcp.types
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StdioTransport, SSETransport, StreamableHttpTransport
from loguru import logger

from ..schema import ToolCall


class FastMcpClient:
    """
    Asynchronous MCP client using FastMCP, supporting stdio and HTTP connections.
    Includes built-in retry logic, timeout handling, and environment injection.
    """

    def __init__(
            self,
            name: str,
            config: dict,
            append_env: bool = False,
            max_retries: int = 3,
            timeout: Optional[float] = None,
    ):
        """
        Initialize the FastMCP client.

        :param name: Unique identifier for the MCP server.
        :param config: Configuration dict (command/args for stdio, or url/type for HTTP).
        :param append_env: If True, merges current process environment with config env.
        :param max_retries: Number of attempts for connection and tool calls.
        :param timeout: Seconds to wait before timing out an operation.
        """
        self.name: str = name
        self.config: dict = config
        self.append_env: bool = append_env
        self.max_retries: int = max_retries
        self.timeout: Optional[float] = timeout

        self.client: Optional[Client] = None
        self._transport = self._create_transport()

    def _create_transport(self):
        """Creates the appropriate transport layer based on the provided configuration."""
        command = self.config.get("command")

        if command:
            # Handle Stdio Transport (Local Processes)
            # Resolve full path for executables like 'npx'
            resolved_command = shutil.which(command) or command

            env_params = os.environ.copy() if self.append_env else {}
            if custom_env := self.config.get("env"):
                env_params.update(custom_env)

            return StdioTransport(
                command=resolved_command,
                args=self.config.get("args", []),
                env=env_params if env_params else None,
                cwd=self.config.get("cwd"),
            )
        else:
            # Handle HTTP Transport (SSE or Streamable HTTP)
            url = self.config.get("url")
            if not url:
                raise ValueError(f"Server config '{self.name}' must contain 'command' or 'url'")

            transport_type = str(self.config.get("type", "")).lower()
            kwargs: dict = {"url": url}

            # Inject environment variables into headers (e.g., Bearer {API_KEY})
            if headers := self.config.get("headers"):
                formatted_headers = {}
                for k, v in headers.items():
                    if isinstance(v, str) and "{" in v:
                        formatted_headers[k] = v.format(**os.environ)
                    else:
                        formatted_headers[k] = v
                kwargs["headers"] = formatted_headers

            if "timeout" in self.config:
                kwargs["sse_read_timeout"] = self.config["timeout"]

            # Selection logic: Explicit type or fallback based on URL suffix
            if transport_type in ["streamable_http", "streamablehttp"]:
                return StreamableHttpTransport(**kwargs)
            elif url.endswith("/sse") or transport_type == "sse":
                return SSETransport(**kwargs)
            else:
                return StreamableHttpTransport(**kwargs)

    async def _execute_with_retry(self, operation: Callable[[], Coroutine], op_name: str) -> Any:
        """
        Internal helper to wrap coroutines with timeout and retry logic (DRY principle).
        """
        last_exception = None

        for i in range(self.max_retries):
            try:
                if self.timeout is not None:
                    return await asyncio.wait_for(operation(), timeout=self.timeout)

                return await operation()

            except asyncio.TimeoutError as _:
                last_exception = TimeoutError(f"{self.name} {op_name} timed out after {self.timeout}s")
                logger.error(f"Attempt {i + 1}/{self.max_retries} failed: {last_exception}")

            except Exception as e:
                last_exception = e
                logger.exception(f"Attempt {i + 1}/{self.max_retries} failed: {self.name} {op_name} -> {e}")

            if i < self.max_retries - 1:
                # Exponential backoff: 1s, 2s, 3s...
                await asyncio.sleep(1 + i)

        raise last_exception if last_exception else RuntimeError(f"Operation {op_name} failed unknown")

    async def __aenter__(self) -> "FastMcpClient":
        """Context manager entry: initializes and starts the MCP client."""

        async def _connect():
            self.client = Client(
                transport=self._transport,
                name=self.name,
                timeout=self.timeout,
            )
            return await self.client.__aenter__()

        await self._execute_with_retry(_connect, "initialization")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: ensures resources are cleaned up."""
        if self.client:
            try:
                # Give a small grace period for cleanup, no retries here to avoid hanging
                await asyncio.wait_for(self.client.__aexit__(exc_type, exc_val, exc_tb), timeout=5.0)
            except Exception as e:
                logger.debug(f"Cleanup error for {self.name}: {e}")
            finally:
                self.client = None

    async def list_tools(self) -> List[mcp.types.Tool]:
        """Fetch all available tools from the server."""
        if not self.client:
            raise RuntimeError(f"Server {self.name} is not connected.")

        return await self._execute_with_retry(
            lambda: self.client.list_tools(),
            "list_tools"
        )

    async def list_tool_calls(self) -> List[ToolCall]:
        """Fetch tools and convert them to the simplified ToolCall format."""
        tools = await self.list_tools()
        return [ToolCall.from_mcp_tool(t) for t in tools]

    async def call_tool(
            self,
            tool_name: str,
            arguments: dict,
            parse_result: bool = False,
    ) -> Union[str, CallToolResult]:
        """
        Execute a specific tool on the server.

        :param tool_name: The name of the tool to invoke.
        :param arguments: The parameters for the tool.
        :param parse_result: If True, returns a concatenated string of text content.
        """
        if not self.client:
            raise RuntimeError(f"Server {self.name} is not connected.")

        result: CallToolResult = await self._execute_with_retry(
            lambda: self.client.call_tool(tool_name, arguments),
            f"call_tool:{tool_name}"
        )

        if parse_result:
            return self._parse_tool_content(result)
        return result

    @staticmethod
    def _parse_tool_content(result: CallToolResult) -> str:
        """Converts complex tool result content into a readable string."""
        text_parts = []
        for block in result.content:
            if isinstance(block, mcp.types.TextContent):
                text_parts.append(block.text)
            elif isinstance(block, mcp.types.ImageContent):
                text_parts.append(f"[Image content: {block.mimeType}]")
            elif hasattr(block, "text"):  # Fallback for duck-typing
                text_parts.append(getattr(block, "text"))

        return "\n".join(text_parts) if text_parts else str(result)
