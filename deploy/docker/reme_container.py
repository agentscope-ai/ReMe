"""Container startup and health checks using ReMe's existing CLI contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from urllib.request import ProxyHandler, Request, build_opener

HEALTH_STATE_PATH = Path("/tmp/reme-health.json")
_ENV_OVERRIDES = {
    "REME_CONFIG": "config",
    "REME_WORKSPACE_DIR": "workspace_dir",
    "REME_HOST": "service.host",
    "REME_PORT": "service.port",
    "REME_TIMEZONE": "timezone",
}


def start_command(arguments: list[str], environment: dict[str, str]) -> list[str]:
    """Add container defaults only to `start`; explicit CLI arguments win."""
    from reme.config import deep_merge_config, parse_kwargs

    defaults = parse_kwargs(
        "log_to_file=false",
        *[f"{key}={json.dumps(environment[name])}" for name, key in _ENV_OVERRIDES.items() if environment.get(name)],
    )
    # Docker environment values are strings; the HTTP service expects an integer port.
    if environment.get("REME_PORT"):
        defaults["service"]["port"] = int(environment["REME_PORT"])
    overrides = deep_merge_config(defaults, parse_kwargs(*arguments))
    return ["reme", "start", *[f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in overrides.items()]]


def write_health_state(command: list[str], state_path: Path = HEALTH_STATE_PATH) -> None:
    """Record only the effective HTTP address, never credentials or user data."""
    from reme.components.service.cli_service import prepare_start_config
    from reme.config import parse_kwargs
    from reme.constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT, normalize_connect_host
    from reme.plugin import resolve_plugin_runtime

    config = resolve_plugin_runtime(prepare_start_config(parse_kwargs(*command[2:]))).config
    service = config.get("service") or {}
    if service.get("backend") != "http":
        return
    host = normalize_connect_host(service.get("host") or REME_DEFAULT_HOST)
    if host == "::":
        host = "::1"
    port = int(service.get("port", REME_DEFAULT_PORT))
    if not 1 <= port <= 65535:
        raise ValueError("service.port must be between 1 and 65535")
    host = f"[{host}]" if ":" in host else host
    with state_path.open("w", encoding="utf-8") as state_file:
        os.chmod(state_path, 0o600)
        json.dump({"url": f"http://{host}:{port}/health_check"}, state_file)


def healthcheck(state_path: Path = HEALTH_STATE_PATH) -> int:
    """Require both a successful Job and a healthy component snapshot."""
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        request = Request(state["url"], data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        # A deployment's outbound proxy must not intercept its local probe.
        with build_opener(ProxyHandler({})).open(request, timeout=4) as response:
            payload = json.load(response)
        healthy = payload.get("metadata", {}).get("health", {}).get("healthy")
        if payload.get("success") is True and healthy is True:
            return 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    print("ReMe HTTP health check failed; inspect the container logs and POST /health_check", file=sys.stderr)
    return 1


def main() -> int:
    """Prepare startup, then replace this process so signals reach ReMe."""
    arguments = sys.argv[1:]
    if arguments == ["--healthcheck"]:
        return healthcheck()

    HEALTH_STATE_PATH.unlink(missing_ok=True)
    Path(os.environ.get("HOME", "/tmp/reme-home")).mkdir(parents=True, exist_ok=True)
    if not arguments:
        arguments = ["start"]
    start_actions = {"start", "-start", "--start"}
    if len(arguments) >= 2 and arguments[0] == "reme" and arguments[1] in start_actions:
        arguments = arguments[1:]
    if arguments[0] in start_actions:
        from reme.utils import load_env

        load_env()
        arguments = start_command(arguments[1:], dict(os.environ))
        write_health_state(arguments)
    os.execvp(arguments[0], arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
