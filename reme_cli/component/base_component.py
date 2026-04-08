"""Base class for components."""

from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """Base class supporting async start/close and async context management."""

    @abstractmethod
    async def start(self) -> None:
        """Start the component asynchronously."""

    @abstractmethod
    async def close(self) -> None:
        """Close the component asynchronously."""
        ...

    async def __aenter__(self) -> "BaseComponent":
        """Enter async context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close()
