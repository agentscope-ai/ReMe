"""ReMe memory management application entry point."""

import asyncio
import socket
import sys
from collections.abc import Iterator, Sequence

import psutil
from yaml import YAMLError

from .core.application import Application
from .constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT, normalize_connect_host
from .components.service.cli_service import prepare_start_config, should_precheck_start
from .config import parse_action, parse_kwargs, resolve_app_config
from .enumeration import ComponentEnum
from .plugin import resolve_plugin_runtime
from .utils.env_utils import load_env

_CLIENT_KWARGS = {"host", "port", "timeout", "transport", "command", "args", "show_metadata"}


def _parse_cli_invocation(argv: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Parse only the grammar shared by every CLI command family."""
    if not argv:
        raise ValueError("No arguments provided")
    return parse_action(argv[0]), tuple(argv[1:])


async def find_reme(host: str, port: int) -> str:
    """Probe host:port. Returns 'reme', 'occupied', or 'free'."""
    from .components.client.http_client import HttpClient

    try:
        async with HttpClient(host=normalize_connect_host(host), port=port, timeout=2.0) as client:
            async for _ in client(action="health_check"):
                break
        return "reme"
    except Exception:
        pass
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
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


def _start_arguments(cmdline: list[str]) -> list[str] | None:
    """Identify a ReMe CLI process, excluding supervisors containing child argv."""
    if not cmdline:
        return None

    def command_name(value: str) -> str:
        return value.replace("\\", "/").rsplit("/", 1)[-1].lower()

    executable = command_name(cmdline[0])
    if executable in {"reme", "reme.exe"}:
        action_index = 1
    elif executable.startswith(("python", "pypy")):
        action_index = next(
            (
                index
                for index in range(2, len(cmdline))
                if command_name(cmdline[index - 1]) in {"reme", "reme.exe", "reme.py", "reme.reme"}
            ),
            len(cmdline),
        )
        if "-c" in cmdline[:action_index]:
            return None
    else:
        return None
    if action_index >= len(cmdline) or cmdline[action_index] not in {"start", "-start", "--start"}:
        return None
    return [argument for argument in cmdline[action_index + 1 :] if "=" in argument]


def _iter_running_configs(*, allow_partial: bool = False) -> Iterator[tuple[int, dict]]:
    """Replay running ReMe configurations and allow startup overrides for port probes."""
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            info = proc.info
            cmdline = info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        arguments = _start_arguments(cmdline)
        if arguments is None:
            continue
        try:
            overrides = parse_kwargs(*arguments)
        except ValueError:
            continue
        if "job" in overrides:
            continue
        try:
            config = resolve_app_config(log_config=False, _get_config_dir=proc.cwd, **overrides)
        except psutil.NoSuchProcess:
            continue
        except (OSError, ValueError, psutil.AccessDenied, YAMLError):
            # An incomplete replay can supply a probe address without selecting a job client.
            if not allow_partial:
                continue
            service = overrides.get("service")
            if not isinstance(service, dict) or not service:
                continue
            config = overrides
        if should_precheck_start(config):
            yield info["pid"], config


def _scan_reme_procs() -> Iterator[tuple[int, str, int]]:
    """Yield running ReMe PIDs and their configured HTTP probe addresses."""
    for pid, config in _iter_running_configs(allow_partial=True):
        service = config.get("service")
        service = service if isinstance(service, dict) else {}
        host = service.get("host") or REME_DEFAULT_HOST
        port_value = str(service.get("port", REME_DEFAULT_PORT))
        port = int(port_value) if port_value.isdigit() else REME_DEFAULT_PORT
        yield pid, host, port


def running_app_config() -> dict | None:
    """Return a running ReMe's application config, including enabled plugins."""
    return next((config for _, config in _iter_running_configs()), None)


async def locate_reme() -> tuple[str, int, int | None] | None:
    """Find a running reme: try default port, then scanned processes."""
    if await find_reme(REME_DEFAULT_HOST, REME_DEFAULT_PORT) == "reme":
        return REME_DEFAULT_HOST, REME_DEFAULT_PORT, _pid_on_port(REME_DEFAULT_PORT)
    for pid, host, port in _scan_reme_procs():
        if await find_reme(host, port) == "reme":
            return normalize_connect_host(host), port, pid
    return None


async def call_server(job_name: str, /, **kwargs):
    """Call the running server with a client matching its *actual* service config.

    The client backend and its transport/host/port are taken from the running
    ``reme start`` process — its start args replayed through the same
    ``resolve_app_config`` the server used — so a bare ``reme <action>`` reaches
    the server however it was actually started (``http`` REST or ``mcp``
    streamable-http / sse / stdio), including ``service.*`` overrides that never
    touched the on-disk config file. Falls back to local config resolution when
    no running config can be replayed. Explicit ``backend=`` / ``transport=`` / ``host=`` /
    ``port=`` kwargs still win and never leak into the tool payload.
    """
    # config-selecting keys steer client construction; they are not tool args.
    resolve_kwargs = {}
    if isinstance(kwargs.get("config"), str):
        resolve_kwargs["config"] = kwargs.pop("config")
    if isinstance(kwargs.get("service"), dict):
        resolve_kwargs["service"] = kwargs.pop("service")

    # Prefer the running server's complete config so its enabled client plugins
    # are available even when the caller does not repeat config=<name>.
    app_config = running_app_config()
    if app_config is None:
        app_config = resolve_app_config(log_config=False, **resolve_kwargs)
    runtime = resolve_plugin_runtime(app_config)
    app_config = runtime.config
    service = app_config.get("service")
    service = service if isinstance(service, dict) else {}

    service_backend = service.get("backend") or "http"
    backend: str = kwargs.pop("backend", None) or service_backend
    # Seed client kwargs from the service config only when we are actually using
    # that service's backend — transport/host/port are backend-specific, so an
    # explicit backend override must not inherit the other backend's settings.
    seed = service if backend == service_backend else {}
    client_kwargs = {k: seed[k] for k in _CLIENT_KWARGS if k in seed}
    client_kwargs.update({key: kwargs.pop(key) for key in list(kwargs) if key in _CLIENT_KWARGS})

    client_cls = runtime.registry.get(ComponentEnum.CLIENT, backend)
    if client_cls is None:
        raise ValueError(f"Unknown client backend: {backend!r}")
    async with client_cls(**client_kwargs) as client:
        async for chunk in client.call_job(job_name, kwargs):
            print(chunk, end="", flush=True)
        print()


def _run_plugin_command(argv: Sequence[str]) -> None:
    """Run local package management without initializing the application."""
    from .plugin.cli import plugin_cli

    status = plugin_cli(argv)
    if status:
        raise SystemExit(status)


def precheck_start(svc_config: dict | None) -> bool:
    """Pre-flight check for `start`: False if reme is up, exits 1 on port conflict."""
    host = (svc_config or {}).get("host") or REME_DEFAULT_HOST
    port = (svc_config or {}).get("port") or REME_DEFAULT_PORT
    port = int(port)
    status = asyncio.run(find_reme(host, port))
    if status == "reme":
        print(f"reme already running at {normalize_connect_host(host)}:{port}")
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


def _start_application(kwargs: dict, environment: dict) -> None:
    """Resolve startup configuration and run the application."""
    kwargs = prepare_start_config(kwargs)
    kwargs["environment"] = environment
    if should_precheck_start(kwargs) and not precheck_start(kwargs.get("service")):
        return
    Application(**kwargs).run_app()


def main() -> None:
    """Parse CLI arguments and launch the appropriate mode."""
    action, arguments = _parse_cli_invocation(sys.argv[1:])

    if action == "plugins":
        # Package management is local-only and must not load application config,
        # environment files, or a running service.
        _run_plugin_command(arguments)
        return

    environment = load_env()
    kwargs = parse_kwargs(*arguments)
    if action == "start":
        _start_application(kwargs, environment)
    elif action == "find_reme":
        cli_find_reme()
    else:
        asyncio.run(call_server(action, **kwargs))


if __name__ == "__main__":
    main()
