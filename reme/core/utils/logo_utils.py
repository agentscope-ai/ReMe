"""Terminal branding and configuration display utilities."""

import importlib.metadata
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from ..schema import ServiceConfig


def get_version(package_name: str) -> str:
    """Return the installed version of a package or 'unknown'."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def print_logo(service_config: "ServiceConfig"):
    """Print a stylized ASCII logo and service metadata to the console."""
    ascii_art = [
        r" ██████╗  ███████╗ ███╗   ███╗ ███████╗ ",
        r" ██╔══██╗ ██╔════╝ ████╗ ████║ ██╔════╝ ",
        r" ██████╔╝ █████╗   ██╔████╔██║ █████╗   ",
        r" ██╔══██╗ ██╔══╝   ██║╚██╔╝██║ ██╔══╝   ",
        r" ██║  ██║ ███████╗ ██║ ╚═╝ ██║ ███████╗ ",
        r" ╚═╝  ╚═╝ ╚══════╝ ╚═╝     ╚═╝ ╚══════╝ ",
    ]

    start_color = (85, 239, 196)
    end_color = (162, 155, 254)

    logo_text = Text()
    for line in ascii_art:
        line_len = max(1, len(line) - 1)
        for i, char in enumerate(line):
            # Calculate gradient shift per character
            ratio = i / line_len
            rgb = tuple(int(s + (e - s) * ratio) for s, e in zip(start_color, end_color))
            logo_text.append(char, style=f"bold rgb({rgb[0]},{rgb[1]},{rgb[2]})")
        logo_text.append("\n")

    # Layout configuration info
    info_table = Table.grid(padding=(0, 1))
    info_table.add_column(style="bold", justify="center")
    info_table.add_column(style="bold cyan", justify="left")
    info_table.add_column(style="white", justify="left")

    # Add core service info
    info_table.add_row("📦", "Backend:", service_config.backend)

    match service_config.backend:
        case "http":
            host, port = service_config.http.host, service_config.http.port
            info_table.add_row("🔗", "URL:", f"http://{host}:{port}")
            info_table.add_row("📚", "FastAPI:", Text(get_version("fastapi"), style="dim"))
        case "mcp":
            mcp = service_config.mcp
            transport = mcp.transport if mcp.transport else "stdio"
            info_table.add_row("🚌", "Transport:", transport)
            if transport != "stdio":
                url = f"http://{mcp.host}:{mcp.port}"
                if transport == "sse":
                    url += "/sse"
                info_table.add_row("🔗", "URL:", url)
            info_table.add_row("📚", "FastMCP:", Text(get_version("fastmcp"), style="dim"))

    info_table.add_row("🚀", "ReMe:", Text(get_version("reme-ai"), style="dim"))

    # Render layout within a panel
    panel = Panel(
        Group(logo_text, info_table),
        title=service_config.app_name,
        title_align="left",
        border_style="dim",
        padding=(1, 4),
        expand=False,
    )

    # use justify="center" to adjust position
    Console().print(Group("\n", panel, "\n"))
