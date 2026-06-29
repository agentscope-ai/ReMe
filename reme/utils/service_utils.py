"""Service discovery utilities."""

import asyncio
import socket
import sys

import psutil

from ..constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT


async def find_reme(host: str, port: int) -> str:
    """Probe host:port. Returns 'reme', 'occupied', or 'free'."""
    from ..components.client.http_client import HttpClient

    try:
        async with HttpClient(host=host, port=port, timeout=2.0) as client:
            async for _ in client(action="health_check"):
                break
        return "reme"
    except Exception:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return "free"
        except OSError:
            return "occupied"


def _pid_on_port(port: int) -> int | None:
    """PID listening on TCP ``port``, or None. Cross-platform via psutil.

    Iterates per-process rather than calling the system-wide
    ``psutil.net_connections()`` — the latter needs root on macOS, while
    per-process connection enumeration works without elevation for
    processes the current user owns (which reme's own server always is).
    """
    for proc in psutil.process_iter(["pid"]):
        try:
            for conn in proc.net_connections(kind="tcp"):
                if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
                    return proc.info["pid"]
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return None


def _scan_reme_procs() -> list[tuple[int, str, int]]:
    """List running 'reme ... start' processes as (pid, host, port)."""
    procs: list[tuple[int, str, int]] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        # Match a `reme ... start` invocation (mirrors the old `pgrep -af`).
        if "start" not in cmdline or not any("reme" in tok for tok in cmdline):
            continue
        host, port = REME_DEFAULT_HOST, REME_DEFAULT_PORT
        for t in cmdline:
            if t.startswith("service.host="):
                host = t.split("=", 1)[1]
            elif t.startswith("service.port=") and t.split("=", 1)[1].isdigit():
                port = int(t.split("=", 1)[1])
        procs.append((proc.info["pid"], host, port))
    return procs


async def locate_reme() -> tuple[str, int, int | None] | None:
    """Find a running reme: try default port, then scanned processes."""
    if await find_reme(REME_DEFAULT_HOST, REME_DEFAULT_PORT) == "reme":
        return REME_DEFAULT_HOST, REME_DEFAULT_PORT, _pid_on_port(REME_DEFAULT_PORT)
    for pid, host, port in _scan_reme_procs():
        if await find_reme(host, port) == "reme":
            return host, port, pid
    return None


def precheck_start(svc_config: dict | None) -> bool:
    """Pre-flight check for `start`: False if reme is up, exits 1 on port conflict."""
    host = (svc_config or {}).get("host") or REME_DEFAULT_HOST
    port = (svc_config or {}).get("port") or REME_DEFAULT_PORT
    port = int(port)
    status = asyncio.run(find_reme(host, port))
    if status == "reme":
        print(f"reme already running at {host}:{port}")
        return False
    if status == "occupied":
        print(
            f"port {port} occupied. Start on another port: reme start service.port=<other_port>",
            file=sys.stderr,
        )
        sys.exit(1)
    return True


def cli_find_reme() -> None:
    """Handle `reme find_reme`: print HOST/PORT/PID or a hint to start reme."""
    found = asyncio.run(locate_reme())
    if not found:
        print("reme not started. Try: reme start", file=sys.stderr)
        sys.exit(1)
    host, port, pid = found
    print(f"HOST={host} PORT={port} PID={pid or 'unknown'}")
