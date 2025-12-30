import asyncio
import sys
from contextlib import asynccontextmanager
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import sse_client


class UniversalMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._exit_stack = None

    @asynccontextmanager
    async def connect_to_stdio(self, command: str, args: list[str], env: dict = None):
        """支持 npx, uvx 等本地命令"""
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                yield session

    @asynccontextmanager
    async def connect_to_sse(self, url: str):
        """支持 远程 SSE/HTTP Stream 连接"""
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await session.initialize()
                yield session

    async def list_tools(self):
        if not self.session:
            return "未连接到服务器"
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, arguments: dict):
        if not self.session:
            return "未连接到服务器"
        return await self.session.call_tool(tool_name, arguments)


# --- 使用示例 ---


async def main():
    client = UniversalMCPClient()

    # 场景 1: 使用 npx 启动 Node 版本的 MCP Server
    print("\n--- 尝试使用 npx 连接 ---")
    npx_args = ["-y", "@modelcontextprotocol/server-everything", "echo"]
    try:
        async with client.connect_to_stdio("npx", npx_args) as session:
            tools = await client.list_tools()
            print(f"可用工具: {tools}")
    except Exception as e:
        print(f"npx 连接失败: {e}")

    # 场景 2: 使用 uvx 启动 Python 版本的 MCP Server
    print("\n--- 尝试使用 uvx 连接 ---")
    uvx_args = ["mcp-server-git", "--repository", "."]
    try:
        async with client.connect_to_stdio("uvx", uvx_args) as session:
            tools = await client.list_tools()
            print(f"可用工具: {tools}")
    except Exception as e:
        print(f"uvx 连接失败: {e}")

    # 场景 3: 连接远程 SSE 服务器 (HTTP Stream)
    print("\n--- 尝试使用 SSE 连接远程服务器 ---")
    sse_url = "http://localhost:8000/sse"  # 替换为实际地址
    try:
        async with client.connect_to_sse(sse_url) as session:
            tools = await client.list_tools()
            print(f"远程工具: {tools}")
    except Exception as e:
        print(f"SSE 连接失败 (请检查服务器是否开启): {e}")


if __name__ == "__main__":
    asyncio.run(main())
