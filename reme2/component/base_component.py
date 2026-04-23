"""Base class for components."""

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..enumeration import ComponentEnum
from ..utils import get_logger

if TYPE_CHECKING:
    from .application_context import ApplicationContext


class BaseComponent(ABC):
    """Async lifecycle base class with context manager support.

    Subclasses must implement ``_start`` and ``_close``.
    """

    component_type = ComponentEnum.BASE

    def __init__(
            self,
            name: str | None = None,
            backend: str = "",
            app_context: "ApplicationContext | None" = None,
            **kwargs,
    ) -> None:
        self.name: str = name or self.__class__.__name__
        self.backend: str = backend
        self.app_context: "ApplicationContext | None" = app_context
        self.kwargs: dict = dict(kwargs)
        self.logger = get_logger()
        if hasattr(self.logger, "bind"):
            self.logger = self.logger.bind(component=self.name)

        self._is_started: bool = False
        self._lock: asyncio.Lock = asyncio.Lock()

    @abstractmethod
    async def _start(self) -> None:
        """Start the component."""

    @abstractmethod
    async def _close(self) -> None:
        """Close the component."""

    async def start(self) -> None:
        """Start the component. No-op if already started."""
        async with self._lock:
            if self._is_started:
                return
            await self._start()
            self._is_started = True

    async def close(self) -> None:
        """Close the component. No-op if not started."""
        async with self._lock:
            if not self._is_started:
                return
            await self._close()
            self._is_started = False

    async def restart(self) -> None:
        """Close then start."""
        await self.close()
        await self.start()

    @property
    def is_started(self) -> bool:
        return self._is_started

    async def __call__(self, **kwargs):
        raise NotImplementedError

    async def __aenter__(self) -> "BaseComponent":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
