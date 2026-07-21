"""Base class for services that expose jobs over a network protocol."""

import json
import os
from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from ..base_component import BaseComponent
from ..job.base_job import BaseJob
from ...constants import REME_SERVICE_INFO
from ...enumeration import ComponentEnum

if TYPE_CHECKING:
    from ...application import Application


class BaseService(BaseComponent):
    """Skeleton for services (HTTP, MCP, ...) that turn jobs into endpoints or tools."""

    component_type = ComponentEnum.SERVICE

    def __init__(self, jobs: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.jobs: set[str] | None = set(jobs) if jobs is not None else None
        # Underlying framework instance (FastAPI, FastMCP, ...); populated by build_service().
        self.service = None

    # ----- Subclass contract ---------------------------------------------

    @abstractmethod
    def build_service(self, app: "Application") -> None:
        """Instantiate and configure the underlying server framework."""

    @abstractmethod
    def add_job(self, job: BaseJob) -> bool:
        """Register a single job as a callable endpoint or tool.

        Returns True when the job is exposed, False when the service intentionally
        skips it (for example, unsupported job types).
        """

    @abstractmethod
    def start_service(self, app: "Application") -> None:
        """Block on serving requests until shutdown."""

    # ----- Shared helpers ------------------------------------------------

    def _lifespan(self, app: "Application", host: str, port: int):
        """Build an async-context lifespan that brackets the server with app start/close.

        Publishes the bound address via the REME_SERVICE_INFO environment variable so
        in-process clients can discover where this service is listening.
        """

        @asynccontextmanager
        async def lifespan(_):
            await app.start()
            try:
                service_info = json.dumps({"host": host, "port": port})
                os.environ[REME_SERVICE_INFO] = service_info
                self.logger.info(f"{self.name} started: {REME_SERVICE_INFO}={service_info}")
                yield
            finally:
                await app.close()

        return lifespan

    def add_jobs(self, app: "Application") -> None:
        """Register service-enabled jobs, optionally restricted by the configured whitelist."""
        if self.jobs is not None:
            missing = sorted(self.jobs.difference(app.context.jobs))
            if missing:
                raise KeyError(f"Service jobs not found: {', '.join(missing)}")
            disabled = sorted(name for name in self.jobs if not app.context.jobs[name].enable_serve)
            if disabled:
                raise ValueError(f"Service jobs are not enabled for serving: {', '.join(disabled)}")

        for name, job in app.context.jobs.items():
            if not job.enable_serve or (self.jobs is not None and name not in self.jobs):
                continue
            try:
                added = self.add_job(job)
            except Exception as e:
                if self.jobs is not None:
                    raise
                self.logger.error(f"Failed to add job {name}: {e}")
                continue
            if added:
                self.logger.info(f"Added job: {name}")
            elif self.jobs is not None:
                raise TypeError(f"Service does not support job: {name}")
            else:
                self.logger.warning(f"Skipped job: {name}")

    def run_app(self, app: "Application") -> None:
        """Build the service, register jobs, then start serving (blocking)."""
        self.build_service(app)
        self.add_jobs(app)
        self.start_service(app)
