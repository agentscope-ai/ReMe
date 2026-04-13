"""Base class for components."""

from abc import ABC, abstractmethod

from ..enumeration import ComponentEnum
from ..utils.logger_utils import get_logger


class BaseComponent(ABC):
    """Base class for all application components.

    Provides an asynchronous lifecycle with start/close operations and
    async context manager support. State tracking prevents duplicate
    start or close calls.

    Subclasses must implement ``_start`` and ``_close`` to define their
    specific initialization and teardown logic.

    Examples:
        Direct usage::

            comp = MyComponent()
            await comp.start()
            # ... use component ...
            await comp.close()

        Context manager usage::

            async with MyComponent() as comp:
                # ... use component ...

    Attributes:
        component_type: The type identifier for this component, used during
            registry lookup. Defaults to ``ComponentEnum.BASE``.
        _is_started: Internal flag indicating whether the component has been
            started and not yet closed.
    """
    from .application_context import ApplicationContext

    component_type = ComponentEnum.BASE

    def __init__(self, **kwargs) -> None:
        """Initialize a component instance.

        Sets up the component's internal state, binds a structured logger
        with the component's class name, and stores any additional keyword
        arguments for downstream use by subclasses.

        Args:
            **kwargs: Arbitrary keyword arguments forwarded to the component
                subclass. Typically provided by the registry when the
                component is instantiated from configuration.
        """
        self.kwargs: dict = dict(kwargs)
        self.logger = get_logger()
        if hasattr(self.logger, "bind"):
            self.logger = self.logger.bind(component=self.__class__.__name__)
        self._is_started: bool = False

    @abstractmethod
    async def _start(self, app_context: ApplicationContext | None = None) -> None:
        """Perform the actual initialization logic for this component.

        Subclasses must implement this method to set up resources such as
        connections, caches, or background tasks. This method is called
        internally by ``start()`` after verifying the component is not
        already started.

        Args:
            app_context: The shared application context that provides access
                to other initialized components and the application service.
                May be ``None`` if the component does not require cross-component
                references.

        Raises:
            ValueError: If required configuration or dependencies are missing
                or invalid.
            Exception: Any exception raised during resource acquisition will
                propagate to the caller of ``start()``.
        """

    @abstractmethod
    async def _close(self) -> None:
        """Perform the actual teardown logic for this component.

        Subclasses must implement this method to release resources such as
        closing connections, flushing buffers, or cancelling background tasks.
        This method is called internally by ``close()`` after verifying the
        component is in a started state.

        Raises:
            ValueError: If the component is in an unexpected state during shutdown.
            Exception: Any exception raised during resource cleanup will
                propagate to the caller of ``close()``.
        """

    async def start(self, app_context: ApplicationContext | None = None) -> None:
        """Start the component and transition it to an active state.

        This is the public entry point for component initialization. It guards
        against duplicate starts by returning immediately if the component is
        already running, then delegates to ``_start`` for the subclass-specific
        setup.

        Args:
            app_context: The shared application context to pass to ``_start``.

        Raises:
            ValueError: If the component configuration is invalid or required
                dependencies are unavailable (raised by the subclass ``_start``).
        """
        if self._is_started:
            return
        await self._start(app_context)
        self._is_started = True

    async def close(self) -> None:
        """Close the component and release its resources.

        This is the public entry point for component teardown. It guards
        against redundant closes by returning immediately if the component
        has not been started or is already closed, then delegates to ``_close``
        for the subclass-specific cleanup.

        Raises:
            ValueError: If the component is in an inconsistent state that
                prevents safe shutdown (raised by the subclass ``_close``).
        """
        if not self._is_started:
            return
        try:
            await self._close()
        finally:
            self._is_started = False

    async def restart(self, app_context: ApplicationContext | None = None) -> None:
        """Restart the component by closing and then starting it again.

        This method safely tears down the component if it is currently running
        and reinitialized it. If the component is not started, it will simply
        be started.

        If either the close or start operation fails, the exception propagates
        immediately and the component will be left in a non-started state.

        Args:
            app_context: The shared application context to pass during startup.

        Raises:
            ValueError: If the component cannot be cleanly shut down or
                reinitialized due to invalid state or configuration.
        """
        await self.close()
        await self.start(app_context)

    @property
    def is_started(self) -> bool:
        """Return whether the component is currently in a started state.

        Returns:
            ``True`` if ``start()`` has been called and ``close()`` has not
            been called since; ``False`` otherwise.
        """
        return self._is_started

    async def __aenter__(self) -> "BaseComponent":
        """Enter the async context manager by starting the component.

        Returns:
            The component instance, allowing it to be bound in an ``async with``
            statement.

        Raises:
            ValueError: If the component fails to start due to invalid
                configuration or missing dependencies.
        """
        await self.start()
        return self

    async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb,
    ) -> bool:
        """Exit the async context manager by closing the component.

        Any exception raised within the context block is not suppressed.
        If both the context block and ``close()`` raise exceptions, the
        original exception from the context block is preserved and the
        close exception is attached as its ``__cause__`` to maintain
        the full error chain.

        Args:
            exc_type: The exception type if an exception was raised in the
                context block, otherwise ``None``.
            exc_val: The exception value if an exception was raised.
            exc_tb: The traceback if an exception was raised.

        Returns:
            ``False`` to indicate that exceptions should not be suppressed.
        """
        if self._is_started:
            if exc_val is not None:
                try:
                    await self._close()
                except BaseException as close_exc:
                    raise close_exc from exc_val
                finally:
                    self._is_started = False
            else:
                await self.close()
        return False
