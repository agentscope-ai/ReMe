"""Base client abstraction."""

import json
from abc import abstractmethod
from collections.abc import AsyncGenerator

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum


class BaseClient(BaseComponent):
    """Abstract base for clients that communicate with ReMe services."""

    component_type = ComponentEnum.CLIENT

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = None

    async def _start(self) -> None:
        """Initialize the client."""

    async def _close(self) -> None:
        """Close the client and release resources."""

    @abstractmethod
    def _execute(self, action: str, payload: dict) -> AsyncGenerator[str, None]:
        """Backend-specific execution; yield text chunks (single yield for non-streaming backends)."""

    @abstractmethod
    async def list_actions(self) -> list[dict]:
        """Discover available actions on the server; each dict is the raw backend descriptor."""

    async def __call__(self, action: str, **kwargs) -> AsyncGenerator[str, None]:
        """Dispatch: action='list' returns the action catalog; otherwise delegate to _execute()."""
        if action == "list":
            actions = await self.list_actions()
            yield json.dumps(actions, indent=2, ensure_ascii=False)
            return
        async for chunk in self._execute(action, kwargs):
            yield chunk
