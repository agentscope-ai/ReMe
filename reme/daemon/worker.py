"""Private child entry point with readiness reporting and parent-owned shutdown."""

import asyncio
import json
import os
import socket
import sys
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from cryptography.fernet import Fernet

from ..core.application import Application
from ..enumeration import ComponentEnum
from ..plugin import resolve_plugin_runtime
from ..schema import ApplicationConfig
from .config import Settings, resolve_worker_config
from .storage import write_private


def _report(path: Path, **data) -> None:
    write_private(path, json.dumps(data))


def _check_backends(config: dict) -> None:
    runtime = resolve_plugin_runtime(config)
    settings = ApplicationConfig.model_validate(runtime.config)
    groups = [(ComponentEnum.SERVICE, [settings.service]), (ComponentEnum.JOB, settings.jobs.values())]
    groups.extend((ctype, group.values()) for ctype, group in settings.components.items())
    groups.append(
        (
            ComponentEnum.STEP,
            [step for job in settings.jobs.values() if job.enabled for step in (*job.init_steps, *job.steps)],
        ),
    )
    for ctype, group in groups:
        for component in group:
            if not runtime.registry.get(ctype, component.backend):
                raise ValueError(f"Unregistered {ctype} backend: {component.backend}")


async def _serve(config: dict, view: dict, result: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        http = config["service"]
        http.update(host="127.0.0.1", port=port, web_enabled=False)
        for key in tuple(http):
            if key.startswith("ssl_") or key in {"workers", "reload", "fd", "uds"}:
                http.pop(key)
        app = Application(**config)
        service = app.context.service
        service.build_service(app)
        service.add_jobs(app)
        server = uvicorn.Server(
            uvicorn.Config(service.service, host="127.0.0.1", port=port, timeout_graceful_shutdown=5),
        )

        loop = asyncio.get_running_loop()

        def wait_for_parent():
            os.read(sys.stdin.fileno(), 1)
            if not loop.is_closed():
                loop.call_soon_threadsafe(setattr, server, "should_exit", True)

        threading.Thread(target=wait_for_parent, daemon=True).start()
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        while not serving.done() and not server.started:
            await asyncio.sleep(0.05)
        if server.started:
            _report(result, port=port, view=view)
        await serving
        if not server.started:
            raise RuntimeError("ReMe application startup failed. See the service logs.")


async def _test_model(config: dict) -> None:
    """Test the configured provider with the same wrapper in a temporary workspace."""
    components = config["components"]
    wrapper = components["agent_wrapper"]["default"]
    with TemporaryDirectory(prefix="reme-model-test-") as workspace:
        test_components = {
            "agent_wrapper": {"default": {**wrapper, "project_path": workspace, "cwd": workspace, "skills": []}},
        }
        if wrapper["backend"] == "codex":
            server_name = wrapper.get("app_server", "default")
            server = components["codex_app_server"][server_name]
            test_components["codex_app_server"] = {
                server_name: {**server, "cwd": workspace, "codex_home": ".codex"},
            }
        elif wrapper["backend"] == "claude_code":
            test_components["agent_wrapper"]["default"].update(tools=[], mcp_servers={}, setting_sources=[])
            if relay_name := wrapper.get("anthropic_relay"):
                test_components["anthropic_relay"] = {relay_name: components["anthropic_relay"][relay_name]}
        else:
            raise ValueError("This ReMe configuration does not use a managed model provider")
        app = Application(
            **{
                **config,
                "workspace_dir": workspace,
                "enable_logo": False,
                "log_to_console": False,
                "log_to_file": False,
                "jobs": {},
                "environment": {
                    **config.get("environment", {}),
                    "CLAUDE_CONFIG_DIR": str(Path(workspace) / "claude_config"),
                },
                "components": test_components,
            },
        )
        async with app:
            reply = await app.context.components["agent_wrapper"]["default"].reply("Reply with OK.", job_tools=[])
            if reply["status"] != "completed" or not reply["result"]:
                raise RuntimeError("The selected model did not complete the test request")


def main() -> None:
    """Read launch inputs from the parent pipe rather than command line arguments."""
    request = json.loads(sys.stdin.buffer.readline())
    result = Path(request["result"])
    try:
        settings = Settings(**request["settings"]) if request["settings"] is not None else None
        config, settings, view = resolve_worker_config(request["arguments"], settings)
        os.environ.pop("REME_ENCRYPTION_KEY", None)
        if Path(request["directory"]).is_relative_to(Path(view["workspace_dir"])):
            raise ValueError("The supervisor directory must be outside the workspace")
        if request["mode"] == "import":
            _check_backends(config)
            encrypted = Fernet(request["report_key"].encode("ascii")).encrypt(
                json.dumps(settings.private_values()).encode("utf-8"),
            )
            _report(result, view=view, settings=encrypted.decode("ascii"))
        elif request["mode"] == "check":
            _check_backends(config)
            _report(result, view=view)
        elif request["mode"] == "test":
            if not view["model_editable"]:
                raise ValueError("This ReMe configuration does not use a managed model provider")
            if any(field in view["missing_fields"] for field in ("base_url", "api_key", "model")):
                raise ValueError("Complete the model settings before testing the connection")
            asyncio.run(_test_model(config))
            _report(result, view=view)
        else:
            asyncio.run(_serve(config, view, result))
    except Exception as error:  # pylint: disable=broad-except
        if not result.exists():
            # Typed validation and provider errors can contain credentials or other startup inputs.
            message = str(error) if type(error) is ValueError else type(error).__name__
            if isinstance(error, FileExistsError):
                message = "A required workspace directory is occupied by a file"
            for key in ("api_key", "dashscope_api_key"):
                secret = (request.get("settings") or {}).get(key)
                if secret:
                    message = message.replace(secret, "***")
            _report(result, error=message)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
