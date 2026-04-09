"""Base class for components."""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import TYPE_CHECKING

from ..enumeration import ComponentEnum
from ..utils.logger_utils import get_logger

if TYPE_CHECKING:
    from .application_context import ApplicationContext


class BaseComponent(ABC):
    """Base class supporting async start/close and async context management.

    Provides lifecycle management with state tracking to prevent duplicate
    start/close operations.

    Attributes:
        component_type: The type identifier for this component.
        _is_started: Internal flag tracking whether the component has been started.
    """

    component_type = ComponentEnum.BASE

    def __init__(self, **kwargs) -> None:
        """Initialize the component with default state."""
        self.kwargs: dict = kwargs
        self.logger = get_logger()
        if hasattr(self.logger, "bind"):
            self.logger = self.logger.bind(component=self.__class__.__name__)
        self._is_started: bool = False

    @abstractmethod
    async def _start(self, app_context: ApplicationContext | None = None) -> None:
        """Internal method to perform the actual start logic.

        Subclasses should implement this instead of start().
        """

    @abstractmethod
    async def _close(self) -> None:
        """Internal method to perform the actual close logic.

        Subclasses should implement this instead of close().
        """

    async def start(self, app_context: ApplicationContext | None = None) -> None:
        """Start the component asynchronously.

        Does nothing if already started.
        """
        if self._is_started:
            return
        await self._start(app_context)
        self._is_started = True

    async def close(self) -> None:
        """Close the component asynchronously.

        Does nothing if not started or already closed.
        """
        if not self._is_started:
            return
        await self._close()
        self._is_started = False

    async def restart(self, app_context: ApplicationContext | None = None) -> None:
        """Restart the component by closing and then starting again."""
        await self.close()
        await self.start(app_context)

    @property
    def is_started(self) -> bool:
        """Check if the component is currently started."""
        return self._is_started

    async def __aenter__(self) -> "BaseComponent":
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
    ) -> bool:
        """Exit async context manager."""
        await self.close()
        return False
