"""Startup banner with ASCII logo and service metadata."""

import colorsys
import importlib.metadata
import random
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT

if TYPE_CHECKING:
    from ..components.service import BaseService
    from ..schema import ApplicationConfig


def get_version(package_name: str) -> str:
    """Return installed package version, or empty string if not installed."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _hsv_rgb(h: float, s: float = 0.85, v: float = 0.98) -> tuple[int, int, int]:
    """HSV → 0-255 RGB tuple. High saturation+value keeps colors vibrant."""
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def print_logo(app_config: "ApplicationConfig", runtime_service: "BaseService | None" = None):
    """Print rainbow ASCII logo and runtime config (backend, URL, versions).

    Color: each startup picks a random hue rotation; both horizontal
    (across each line) and vertical (line-to-line) sweep ~half the
    hue wheel, so the banner shows a fresh multi-color rainbow gradient
    every run.
    """
    ascii_art = [
        r" ██████╗  ███████╗ ███╗   ███╗ ███████╗ ",
        r" ██╔══██╗ ██╔════╝ ████╗ ████║ ██╔════╝ ",
        r" ██████╔╝ █████╗   ██╔████╔██║ █████╗   ",
        r" ██╔══██╗ ██╔══╝   ██║╚██╔╝██║ ██╔══╝   ",
        r" ██║  ██║ ███████╗ ██║ ╚═╝ ██║ ███████╗ ",
        r" ╚═╝  ╚═╝ ╚══════╝ ╚═╝     ╚═╝ ╚══════╝ ",
    ]

    hue_base = random.random()  # random starting hue per startup
    horizontal_span = 0.5  # half the wheel left-to-right
    vertical_shift = 0.08  # small per-line nudge for 2D rainbow

    logo_text = Text()
    for line_idx, line in enumerate(ascii_art):
        line_len = max(1, len(line) - 1)
        line_hue_start = hue_base + line_idx * vertical_shift
        for i, char in enumerate(line):
            ratio = i / line_len
            r, g, b = _hsv_rgb(line_hue_start + horizontal_span * ratio)
            logo_text.append(char, style=f"bold rgb({r},{g},{b})")
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
            host = getattr(runtime_service, "host", extra.get("host", REME_DEFAULT_HOST))
            port = getattr(runtime_service, "port", extra.get("port", REME_DEFAULT_PORT))
            info_table.add_row("🔗", "URL:", f"http://{host}:{port}")
            mcp_enabled = getattr(runtime_service, "mcp_enabled", extra.get("mcp_enabled", True))
            if mcp_enabled:
                mcp_path = getattr(runtime_service, "mcp_path", extra.get("mcp_path", "/mcp"))
                info_table.add_row("🚌", "MCP:", f"http://{host}:{port}{mcp_path}")
            info_table.add_row("📚", "FastAPI:", Text(get_version("fastapi"), style="dim"))
            if mcp_enabled:
                info_table.add_row("📚", "FastMCP:", Text(get_version("fastmcp"), style="dim"))
        case "mcp":
            transport = getattr(runtime_service, "transport", extra.get("transport", "sse"))
            info_table.add_row("🚌", "Transport:", transport)
            if transport != "stdio":
                host = getattr(runtime_service, "host", extra.get("host", REME_DEFAULT_HOST))
                port = getattr(runtime_service, "port", extra.get("port", REME_DEFAULT_PORT))
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
