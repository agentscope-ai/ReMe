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

        from .component_registry import R
        from .base_component import BaseComponent
        from .job.base_job import BaseJob

        # Initialize the service
        service_config = self.app_config.service
        if not service_config.backend:
            raise ValueError("Service configuration is missing the required 'backend' field")
        service_cls = R.get(ComponentEnum.SERVICE, service_config.backend)
        if not service_cls:
            raise ValueError(
                f"Service references an unregistered backend '{service_config.backend}' "
                f"of type '{ComponentEnum.SERVICE}'"
            )
        self.service = service_cls(**service_config.model_dump(exclude={"backend"}))

        # Initialize all components grouped by type and name
        self.components: dict[ComponentEnum, dict[str, BaseComponent]] = {}
        for component_type, component_configs in self.app_config.components.items():
            self.components[component_type] = {}
            for name, config in component_configs.items():
                if not config.backend:
                    raise ValueError(f"Component '{name}' is missing the required 'backend' field")
                backend_cls = R.get(component_type, config.backend)
                if not backend_cls:
                    raise ValueError(
                        f"Component '{name}' references an unregistered backend '{config.backend}' "
                        f"of type '{component_type}'"
                    )
                self.components[component_type][name] = backend_cls(**config.model_dump(exclude={"backend"}))

        # Initialize all jobs
        self.jobs: dict[str, BaseJob] = {}
        for job_config in self.app_config.jobs:
            if not job_config.backend:
                raise ValueError(f"Job '{job_config.name}' is missing the required 'backend' field")

            job_cls = R.get(ComponentEnum.JOB, job_config.backend)
            if not job_cls:
                raise ValueError(
                    f"Job '{job_config.name}' references an unregistered backend '{job_config.backend}' "
                    f"of type '{ComponentEnum.JOB}'"
                )
            self.jobs[job_config.name] = job_cls(**job_config.model_dump(exclude={"backend"}))
