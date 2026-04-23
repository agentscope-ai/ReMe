"""Abstract base class for service implementations."""

from abc import abstractmethod
from typing import TYPE_CHECKING

from ..base_component import BaseComponent
from ..job.base_job import BaseJob
from ...enumeration import ComponentEnum

if TYPE_CHECKING:
    from ...application import Application


class BaseService(BaseComponent):
    """Abstract base class for services that expose jobs (HTTP, MCP, etc.)."""

    component_type = ComponentEnum.SERVICE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = None

    @abstractmethod
    def build_service(self, app: "Application") -> None:
        ...

    @abstractmethod
    def add_job(self, job: BaseJob) -> None:
        ...

    @abstractmethod
    def start_service(self, app: "Application") -> None:
        ...

    def add_jobs(self, app: "Application") -> None:
        for name, job in app.context.jobs.items():
            try:
                self.add_job(job)
                self.logger.info(f"Successfully Added job {name}")
            except Exception as e:
                self.logger.error(f"Failed to add job {name}: {e}")

    def run_app(self, app: "Application") -> None:
        self.build_service(app)
        self.add_jobs(app)
        self.start_service(app)
