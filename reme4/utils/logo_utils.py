"""Startup banner with ASCII logo and service metadata."""

import importlib.metadata
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from ..schema import ApplicationConfig


def get_version(package_name: str) -> str:
    """Return installed package version, or empty string if not installed."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def print_logo(app_config: "ApplicationConfig"):
    """Print gradient ASCII logo and runtime config (backend, URL, versions)."""
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
            ratio = i / line_len
            rgb = tuple(int(s + (e - s) * ratio) for s, e in zip(start_color, end_color))
            logo_text.append(char, style=f"bold rgb({rgb[0]},{rgb[1]},{rgb[2]})")
        logo_text.append("\n")

    info_table = Table.grid(padding=(0, 1))
    info_table.add_column(style="bold", justify="center")
    info_table.add_column(style="bold cyan", justify="left")
    info_table.add_column(style="white", justify="left")

    # service is a ComponentConfig with extra="allow"; backend-specific fields live in model_extra.
    service = app_config.service
    backend = service.backend
    extra = service.model_extra or {}

    info_table.add_row("📦", "Backend:", backend)

    match backend:
        case "http":
            host = extra.get("host", "localhost")
            port = extra.get("port", 8000)
            info_table.add_row("🔗", "URL:", f"http://{host}:{port}")
            info_table.add_row("📚", "FastAPI:", Text(get_version("fastapi"), style="dim"))
        case "mcp":
            transport = extra.get("transport", "stdio")
            info_table.add_row("🚌", "Transport:", transport)
            if transport != "stdio":
                host = extra.get("host", "localhost")
                port = extra.get("port", 8000)
                url = f"http://{host}:{port}"
                if transport == "sse":
                    url += "/sse"
                info_table.add_row("🔗", "URL:", url)
            info_table.add_row("📚", "FastMCP:", Text(get_version("fastmcp"), style="dim"))

    info_table.add_row("🚀", "ReMe:", Text(get_version("reme-ai"), style="dim"))

    panel = Panel(
        Group(logo_text, info_table),
        title=app_config.app_name,
        title_align="left",
        border_style="dim",
        padding=(1, 4),
        expand=False,
    )

    Console().print(Group("\n", panel, "\n"))
