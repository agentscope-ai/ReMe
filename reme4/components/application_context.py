"""Application context: shared state container for components, jobs, and service."""

from ..enumeration import ComponentEnum
from ..schema import ApplicationConfig


class ApplicationContext:
    """Holds the parsed config and instantiated components, jobs, and service.

    Acts as a passive state container. The actual wiring (resolving backends from
    the registry and instantiating each component) is performed by Application.
    """

    def __init__(self, **kwargs):
        # Parse and validate raw config kwargs into a typed ApplicationConfig.
        self.app_config: ApplicationConfig = ApplicationConfig(**kwargs)

        # Local imports to avoid circular dependencies during module init.
        from .base_component import BaseComponent
        from .job import BaseJob
        from .service import BaseService

        # Service endpoint (e.g. HTTP/MCP). Populated by Application.__init__.
        self.service: BaseService | None = None
        # Components keyed by type then by user-defined name.
        self.components: dict[ComponentEnum, dict[str, BaseComponent]] = {}
        # Jobs keyed by user-defined name.
        self.jobs: dict[str, BaseJob] = {}
