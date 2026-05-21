"""Base service class for exposing jobs via HTTP, MCP, etc."""

from abc import abstractmethod
from typing import TYPE_CHECKING

from ..base_component import BaseComponent
from ..job.base_job import BaseJob
from ...enumeration import ComponentEnum

if TYPE_CHECKING:
    from ...application import Application


class BaseService(BaseComponent):
    """Base class for services that expose jobs via HTTP, MCP, etc."""

    component_type = ComponentEnum.SERVICE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = None

    @abstractmethod
    def build_service(self, app: "Application") -> None:
        """Initialize the underlying service framework."""

    @abstractmethod
    def add_job(self, job: BaseJob) -> None:
        """Register a single job with the service."""

    @abstractmethod
    def start_service(self, app: "Application") -> None:
        """Start serving requests."""

    def add_jobs(self, app: "Application") -> None:
        """Register all non-background jobs from the application context."""
        for name, job in app.context.jobs.items():
            if job.backend == "background":
                continue
            try:
                self.add_job(job)
                self.logger.info(f"Added job: {name}")
            except Exception as e:
                self.logger.error(f"Failed to add job {name}: {e}")

    def run_app(self, app: "Application") -> None:
        """Build, populate, and start the service."""
        self.build_service(app)
        self.add_jobs(app)
        self.start_service(app)
