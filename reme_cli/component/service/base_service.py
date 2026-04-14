"""Abstract base class for service implementations."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ..job.base_job import BaseJob
from ...enumeration import ComponentEnum


class BaseService(BaseComponent):
    """Abstract base class for services that expose jobs.

    Services provide different ways to invoke jobs (HTTP, CLI, MCP, etc.).
    Subclasses must implement add_job to register jobs with the service.
    """
    component_type = ComponentEnum.SERVICE

    from ...application import Application

    def __init__(self, **kwargs):
        """Initialize the service.

        Args:
            **kwargs: Additional service-specific configuration.
        """
        super().__init__(**kwargs)
        self.service = None

    async def _start(self, app_context=None) -> None:
        """Default empty implementation for sync services."""

    async def _close(self) -> None:
        """Default empty implementation for sync services."""

    @abstractmethod
    def add_job(self, job: BaseJob) -> None:
        """Register a job with the service.

        Args:
            job: The job to register.
        """

    @abstractmethod
    def build_service(self, app: "Application") -> None:
        """Build the service.

        Args:
            app: The application instance.
        """

    @abstractmethod
    def start_service(self, app: "Application") -> None:
        """Start the service."""

    def add_jobs(self, app: "Application") -> None:
        for name, job in app.context.jobs.values():
            try:
                self.add_job(job)
                self.logger.info(f"Added job {name}")
            except Exception as e:
                self.logger.error(f"Failed to add job {name}: {e}")

    def run_app(self, app: "Application") -> None:
        """Register all jobs from the application and start the service.

        Args:
            app: The application containing jobs to register.
        """
        self.build_service(app)
        self.add_jobs(app)
        self.start_service(app)
