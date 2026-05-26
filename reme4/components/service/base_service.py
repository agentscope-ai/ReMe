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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Underlying framework instance (FastAPI, FastMCP, ...); populated by build_service().
        self.service = None

    # ----- Subclass contract ---------------------------------------------

    @abstractmethod
    def build_service(self, app: "Application") -> None:
        """Instantiate and configure the underlying server framework."""

    @abstractmethod
    def add_job(self, job: BaseJob) -> None:
        """Register a single job as a callable endpoint or tool."""

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
            service_info = json.dumps({"host": host, "port": port})
            os.environ[REME_SERVICE_INFO] = service_info
            self.logger.info(f"{self.name} started: {REME_SERVICE_INFO}={service_info}")
            yield
            await app.close()

        return lifespan

    def add_jobs(self, app: "Application") -> None:
        """Register every job whose enable_serve flag is True."""
        for name, job in app.context.jobs.items():
            if not job.enable_serve:
                continue
            try:
                self.add_job(job)
                self.logger.info(f"Added job: {name}")
            except Exception as e:
                self.logger.error(f"Failed to add job {name}: {e}")

    def run_app(self, app: "Application") -> None:
        """Build the service, register jobs, then start serving (blocking)."""
        self.build_service(app)
        self.add_jobs(app)
        self.start_service(app)
