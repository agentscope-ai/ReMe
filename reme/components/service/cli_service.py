"""CLI service: run one configured job locally, then exit."""

import asyncio
import json
import sys
from typing import TYPE_CHECKING, Any

from .base_service import BaseService
from ..component_registry import R
from ..job import BaseJob
from ...config import resolve_app_config
from ...schema import ApplicationConfig
from ...utils import get_logger

if TYPE_CHECKING:
    from ...application import Application

_APP_CONFIG_KEYS = set(ApplicationConfig.model_fields)


def prepare_start_config(kwargs: dict) -> dict:
    """Resolve ``reme start`` kwargs, translating top-level ``job=...`` into internal cli service config."""
    if "job" not in kwargs:
        return resolve_app_config(**kwargs)
    return _prepare_job_start_config(dict(kwargs))


def should_precheck_start(config: dict) -> bool:
    """CLI service does not bind a port, so it should skip service port prechecks."""
    service = config.get("service")
    return not (isinstance(service, dict) and service.get("backend") == "cli")


def _prepare_job_start_config(kwargs: dict) -> dict:
    """Translate ``reme start job=...`` args into the internal cli service fields."""
    job = kwargs.pop("job")
    config_kwargs: dict = {}
    job_args: dict = {}

    for key, value in kwargs.items():
        if key == "config" or key in _APP_CONFIG_KEYS:
            config_kwargs[key] = value
        else:
            job_args[key] = value

    # One-shot CLI jobs should print only their answer by default. Reconfigure
    # before resolve_app_config() so even config-loading logs stay off stdout.
    if "log_to_console" not in config_kwargs:
        get_logger(log_to_console=False, log_to_file=False, force_init=True)

    config = resolve_app_config(**config_kwargs)
    if "enable_logo" not in config_kwargs:
        config["enable_logo"] = False
    if "log_to_console" not in config_kwargs:
        config["log_to_console"] = False
    service = dict(config.get("service") or {})
    service.update({"backend": "cli", "job": job, "job_args": job_args})
    config["service"] = service
    return config


@R.register("cli")
class CliService(BaseService):
    """Execute a single job through the normal application lifecycle without serving a port."""

    def __init__(
        self,
        job: str = "",
        job_args: dict[str, Any] | None = None,
        show_metadata: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.job = job
        self.job_args = job_args or {}
        self.show_metadata = show_metadata

    def build_service(self, app: "Application") -> None:
        """No network framework is needed for local CLI execution."""
        self.service = None

    def add_job(self, job: BaseJob) -> bool:
        """CLI execution does not register jobs; Application already owns them."""
        return False

    def start_service(self, app: "Application") -> None:
        """Run the configured job once and print the same human-facing answer style as CLI clients."""
        asyncio.run(self._run_job(app))

    def run_app(self, app: "Application") -> None:
        """Bypass BaseService.add_jobs(), which is only meaningful for serving protocols."""
        self.build_service(app)
        self.start_service(app)

    async def _run_job(self, app: "Application") -> None:
        if not self.job:
            raise ValueError("cli service requires service.job")

        start_job = getattr(app, "start_job", None)
        if start_job is None:
            await app.start()
        else:
            await start_job(self.job)
        try:
            response = await app.run_job(self.job, **self.job_args)
            output = self._format_response(response.answer, response.metadata)
            if response.success:
                print(output)
            else:
                print(output, file=sys.stderr)
                raise SystemExit(1)
        finally:
            await app.close()

    def _format_response(self, answer: Any, metadata: dict | None) -> str:
        if not isinstance(answer, str):
            answer = json.dumps(answer, ensure_ascii=False, indent=2)
        parts = [answer]
        if self.show_metadata and metadata:
            parts.append(json.dumps(metadata, ensure_ascii=False))
        return "\n".join(parts)
