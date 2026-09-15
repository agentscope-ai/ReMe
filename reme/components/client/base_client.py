"""Base client abstraction."""

import json
import os
from abc import abstractmethod
from collections.abc import AsyncGenerator

from ..base_component import BaseComponent
from ...constants import (
    REME_DEFAULT_HOST,
    REME_DEFAULT_PORT,
    REME_SERVICE_INFO,
    normalize_connect_host,
)
from ...enumeration import ComponentEnum


class BaseClient(BaseComponent):
    """Abstract base for clients that communicate with ReMe services."""

    component_type = ComponentEnum.CLIENT

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = None

    def _resolve_service_address(self, host: str | None, port: int | None) -> tuple[str, int]:
        """Resolve explicit, discovered, or default network coordinates."""
        if not (host and port):
            if service_info := os.environ.get(REME_SERVICE_INFO):
                try:
                    data = json.loads(service_info)
                    host, port = data["host"], data["port"]
                except Exception:
                    self.logger.warning(f"Invalid service info: {service_info}")
                    host, port = REME_DEFAULT_HOST, REME_DEFAULT_PORT
            else:
                host, port = REME_DEFAULT_HOST, REME_DEFAULT_PORT
        return normalize_connect_host(host), port

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
