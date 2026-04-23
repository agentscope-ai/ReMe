"""Application context for initializing and managing all configured components."""

from ..enumeration import ComponentEnum
from ..schema import ApplicationConfig


class ApplicationContext:
    """Application context that initializes and manages all configured components.

    This class is responsible for parsing the application configuration,
    resolving backend implementations for each component via the registry,
    and instantiating services, components, and jobs.
    """

    def __init__(self, **kwargs):
        """Initialize the application context from configuration kwargs.

        Args:
            **kwargs: Keyword arguments that form the ApplicationConfig,
                including app_name, service, components, and jobs.

        Raises:
            ValueError: If a required backend is missing for the service, any component, or any job,
                or if a service, component, or job references an unregistered backend type.
        """
        self.app_config: ApplicationConfig = ApplicationConfig(**kwargs)

        from .base_component import BaseComponent
        from .job import BaseJob
        from .service import BaseService

        self.service: BaseService | None = None
        self.components: dict[ComponentEnum, dict[str, BaseComponent]] = {}
        self.jobs: dict[str, BaseJob] = {}
