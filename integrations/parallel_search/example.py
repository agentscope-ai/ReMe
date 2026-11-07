"""Call Parallel's public web-search MCP tools through ReMe's MCP client."""

import argparse
import asyncio
import sys
from urllib.parse import urlparse

from fastmcp.client import StreamableHttpTransport

from reme.components.client import MCPClient

SERVER_URL = "https://mcp.parallel.ai/mcp"
REQUEST_TIMEOUT_SECONDS = 60


def create_client() -> MCPClient:
    """Create a ReMe client for Parallel's anonymous streamable-HTTP endpoint."""
    transport = StreamableHttpTransport(url=SERVER_URL)
    return MCPClient(transport=transport, timeout=REQUEST_TIMEOUT_SECONDS)


def validate_url(value: str) -> str:
    """Accept absolute HTTP(S) URLs without credentials for the remote fetch tool."""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise argparse.ArgumentTypeError("fetch URL must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise argparse.ArgumentTypeError("fetch URL must not include embedded credentials")
    return value


async def run_example(topic: str, fetch_url: str, client: MCPClient, emit=print) -> None:
    """Discover and run the two public search tools without writing to ReMe memory."""
    async with client:
        tools = await client.list_actions()
        available = {tool["name"] for tool in tools}
        required = {"web_search", "web_fetch"}
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(f"Parallel MCP endpoint is missing required tools: {', '.join(missing)}")

        emit(f"## Search: {topic}")
        async for result in client(
            action="web_search",
            objective=topic,
            search_queries=[topic],
        ):
            emit(result)

        emit(f"## Fetch: {fetch_url}")
        async for result in client(action="web_fetch", urls=[fetch_url], objective=topic):
            emit(result)


def main(argv: list[str] | None = None) -> int:
    """Run the opt-in example from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("topic", help="what to search for on the public web")
    parser.add_argument("url", type=validate_url, help="HTTP(S) URL to fetch after searching")
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_example(args.topic, args.url, create_client()))
    except Exception as exc:  # Report provider and transport failures to the caller.
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 429:
            print(
                "Parallel Search rate limit reached. Wait before retrying or see provider docs "
                "for authenticated access.",
                file=sys.stderr,
            )
        else:
            print(f"Parallel MCP request failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
