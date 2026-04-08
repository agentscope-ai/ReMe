"""Base class for components."""

from abc import ABC, abstractmethod

from ..enumeration import ComponentEnum


class BaseComponent(ABC):
    """Base class supporting async start/close and async context management."""

    component_type = ComponentEnum.BASE

    @abstractmethod
    async def start(self) -> None:
        """Start the component asynchronously."""

    @abstractmethod
    async def close(self) -> None:
        """Close the component asynchronously."""

    async def __aenter__(self) -> "BaseComponent":
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit async context manager."""
        await self.close()

        if exc_type is not None:
            return True
        return False
