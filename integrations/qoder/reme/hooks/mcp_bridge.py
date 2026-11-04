#!/usr/bin/env python3
"""Host-managed stdio connection to the user's independently running ReMe service."""

from auto_memory import load_config


def main() -> None:
    """Resolve persistent settings on connection; leave MCP transport handling to FastMCP."""
    config = load_config()
    try:
        from fastmcp.server import create_proxy
        from fastmcp.server.providers.proxy import ProxyClient
    except ImportError as exc:
        raise SystemExit(
            "ReMe MCP requires fastmcp>=3.1 in the host's python3 environment (included in reme-ai).",
        ) from exc
    upstream = ProxyClient(config["mcp_url"], timeout=config["request_timeout"])
    proxy = create_proxy(upstream, name="ReMe")
    proxy.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
