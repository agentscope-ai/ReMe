"""vault MCP server entry point.

Usage:
    python -m reme2.mcp.server [config=path/to/yaml] [service.transport=stdio]
    python reme2/mcp/server.py [config=path/to/yaml] [service.transport=stdio]

By default loads `reme2/config/service.yaml` and starts the
ReMe2 application with the MCP service registered.

Env vars:
    VAULT_PATH — override vault watch path (overrides config + cli args).
"""

import os
import sys
from pathlib import Path

# Make this work whether invoked as `python -m reme2.mcp.server` (repo
# root already on sys.path) or `python reme2/mcp/server.py` (only this
# file's dir is on sys.path — without this, `import reme2.memory` would
# fail with ModuleNotFoundError).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

# Load .env from repo root before anything else touches env vars
# (embedding/llm clients read them at construction time).
load_dotenv(_REPO_ROOT / ".env")

# Eagerly import side-effect modules so all @R.register() decorators run
# before Application introspects the registry.
import reme2  # noqa: E402,F401
import reme2.memory  # noqa: E402,F401  -- registers every agent-facing tool
import reme2.component.service.mcp_service  # noqa: E402,F401

from reme2.application import Application  # noqa: E402
from reme2.config import parse_args  # noqa: E402


_DEFAULT_CONFIG = str(_REPO_ROOT / "reme2" / "config" / "service.yaml")


def main() -> None:
    argv = list(sys.argv[1:])
    if not argv or argv[0].startswith("config=") or "=" in argv[0]:
        argv.insert(0, "start")
    if not any(a.startswith("config=") for a in argv[1:]):
        argv.insert(1, f"config={_DEFAULT_CONFIG}")

    vault_path = os.environ.get("VAULT_PATH")
    if vault_path:
        argv.append(f"components.file_store.default.working_dir={vault_path}")
        argv.append(f"components.file_store.default.db_path={vault_path}/.reme")
        argv.append(f"service.sidecar_info_path={vault_path}/.reme/sidecar.json")

    sidecar_port = os.environ.get("VAULT_HTTP_PORT")
    if sidecar_port:
        argv.append(f"service.sidecar_http_port={sidecar_port}")

    action, config = parse_args(*argv)
    if action != "start":
        raise SystemExit("reme2.mcp.server only supports the 'start' action")

    app = Application(**config)
    app.run_app()


if __name__ == "__main__":
    main()
