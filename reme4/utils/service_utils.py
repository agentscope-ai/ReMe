"""Service discovery utilities."""

import asyncio
import socket
import subprocess
import sys

from ..components.client.http_client import HttpClient
from ..constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT


async def find_reme(host: str, port: int) -> str:
    """Probe host:port. Returns 'reme', 'occupied', or 'free'."""
    try:
        async with HttpClient(action="health_check", host=host, port=port, timeout=2.0) as c:
            await c()
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


def _run(cmd: list[str]) -> str:
    """Run cmd; return stdout, or empty on failure."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _pid_on_port(port: int) -> int | None:
    """Return PID listening on TCP port, or None."""
    out = _run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"]).strip()
    return int(out.splitlines()[0]) if out else None


def _scan_reme_procs() -> list[tuple[int, str, int]]:
    """Find 'reme ... start' processes. Returns [(pid, host, port), ...]."""
    procs = []
    for line in _run(["pgrep", "-af", "reme.* start"]).splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        host, port = REME_DEFAULT_HOST, REME_DEFAULT_PORT
        for t in parts[1:]:
            if t.startswith("service.host="):
                host = t.split("=", 1)[1]
            elif t.startswith("service.port="):
                try:
                    port = int(t.split("=", 1)[1])
                except ValueError:
                    pass
        procs.append((pid, host, port))
    return procs


async def locate_reme() -> tuple[str, int, int | None] | None:
    """Locate a running reme. Returns (host, port, pid) or None."""
    # 1. Try default host:port
    if await find_reme(REME_DEFAULT_HOST, REME_DEFAULT_PORT) == "reme":
        return REME_DEFAULT_HOST, REME_DEFAULT_PORT, _pid_on_port(REME_DEFAULT_PORT)
    # 2. Scan reme processes and probe each
    for pid, host, port in _scan_reme_procs():
        if await find_reme(host, port) == "reme":
            return host, port, pid
    return None


def precheck_start(svc_config: dict | None) -> bool:
    """Pre-flight check before `start`. Returns True if caller should proceed.

    Prints a message and returns False if reme is already running.
    Exits with code 1 if the port is occupied by a non-reme process.
    """
    host = (svc_config or {}).get("host") or REME_DEFAULT_HOST
    port = (svc_config or {}).get("port") or REME_DEFAULT_PORT
    status = asyncio.run(find_reme(host, port))
    if status == "reme":
        print(f"reme already running at {host}:{port}")
        return False
    if status == "occupied":
        print(
            f"port {port} is occupied by another process. "
            f"Start with a different port: reme4 start service.port=<other_port>",
            file=sys.stderr,
        )
        sys.exit(1)
    return True


def cli_find_reme() -> None:
    """Handle the `find_reme` CLI action: print HOST/PORT/PID or a hint."""
    found = asyncio.run(locate_reme())
    if found:
        host, port, pid = found
        print(f"HOST={host} PORT={port} PID={pid if pid is not None else 'unknown'}")
    else:
        print("reme not started. Try: reme start", file=sys.stderr)
        sys.exit(1)
