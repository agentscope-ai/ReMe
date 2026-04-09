from .application_context import ApplicationContext
from .component import BaseComponent


class Application(BaseComponent):
    """Application component for managing the main application."""

    def __init__(self, *args, config: str = "", **kwargs) -> None:
        super().__init__()
        self.context = ApplicationContext(*args, config=config, **kwargs)

    async def _start(self, app_context: ApplicationContext | None = None) -> None:
        """Start the application."""

    async def _close(self) -> None:
        """Close the application."""
