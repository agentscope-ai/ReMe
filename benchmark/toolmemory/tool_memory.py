"""Official-style ReMe Tool Memory HTTP helpers.

Aligned with ReMe Tool Memory HTTP APIs (see ReMe cookbook
``use_tool_memory_demo.py`` and docs under ``docs/tool_memory/``):

- ``add_tool_call_result``
- ``summary_tool_memory``
- ``retrieve_tool_memory``

Response memories are read from ``metadata.memory_list[].content``.
This module does not use ExpG-only fields such as ``no_persist``,
``source_task``, or ``add_to``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8002"


class ToolMemoryFetcher:
    """HTTP client for ReMe Tool Memory endpoints."""

    def __init__(
        self,
        workspace_id: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.workspace_id = workspace_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    @staticmethod
    def _join_tool_names(tool_names: List[str] | str) -> str:
        if isinstance(tool_names, str):
            return tool_names
        return ",".join(tool_names)

    @staticmethod
    def _memory_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            return []
        memory_list = metadata.get("memory_list") or []
        return memory_list if isinstance(memory_list, list) else []

    @classmethod
    def _content_by_tool(cls, payload: Dict[str, Any]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for memory in cls._memory_list(payload):
            if not isinstance(memory, dict):
                continue
            tool_name = str(memory.get("when_to_use") or "").strip()
            content = memory.get("content") or ""
            if tool_name:
                result[tool_name] = str(content)
        return result

    async def add_tool_call_result_async(
        self,
        tool_call_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Call ``add_tool_call_result``."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._url("add_tool_call_result"),
                json={
                    "workspace_id": self.workspace_id,
                    "tool_call_results": tool_call_results,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    async def summary_tool_memory_async(
        self,
        tool_names: List[str] | str,
    ) -> Dict[str, Any]:
        """Call ``summary_tool_memory``."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._url("summary_tool_memory"),
                json={
                    "workspace_id": self.workspace_id,
                    "tool_names": self._join_tool_names(tool_names),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    async def retrieve_tool_memory_async(
        self,
        tool_names: List[str] | str,
    ) -> Dict[str, Any]:
        """Call ``retrieve_tool_memory``."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._url("retrieve_tool_memory"),
                json={
                    "workspace_id": self.workspace_id,
                    "tool_names": self._join_tool_names(tool_names),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    async def collect_memory_async(
        self,
        tool_names: List[str],
    ) -> Dict[str, str]:
        """Summarize then retrieve guidance for tools.

        Returns:
            Mapping from tool name to memory ``content`` string.
        """
        if not tool_names:
            return {}

        names = self._join_tool_names(tool_names)
        try:
            summary = await self.summary_tool_memory_async(names)
            if not summary.get("success"):
                logger.warning("summary_tool_memory failed for %s", names)
        except Exception as exc:  # noqa: BLE001
            logger.warning("summary_tool_memory error for %s: %s", names, exc)

        try:
            retrieved = await self.retrieve_tool_memory_async(names)
        except Exception as exc:  # noqa: BLE001
            logger.warning("retrieve_tool_memory error for %s: %s", names, exc)
            return {}

        if not retrieved.get("success"):
            logger.warning("retrieve_tool_memory failed for %s", names)
            return {}

        return self._content_by_tool(retrieved)

    def add_tool_call_result(
        self,
        tool_call_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Sync wrapper for ``add_tool_call_result``."""
        with httpx.Client() as client:
            response = client.post(
                self._url("add_tool_call_result"),
                json={
                    "workspace_id": self.workspace_id,
                    "tool_call_results": tool_call_results,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    def summary_tool_memory(self, tool_names: List[str] | str) -> Dict[str, Any]:
        """Sync wrapper for ``summary_tool_memory``."""
        with httpx.Client() as client:
            response = client.post(
                self._url("summary_tool_memory"),
                json={
                    "workspace_id": self.workspace_id,
                    "tool_names": self._join_tool_names(tool_names),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    def retrieve_tool_memory(self, tool_names: List[str] | str) -> Dict[str, Any]:
        """Sync wrapper for ``retrieve_tool_memory``."""
        with httpx.Client() as client:
            response = client.post(
                self._url("retrieve_tool_memory"),
                json={
                    "workspace_id": self.workspace_id,
                    "tool_names": self._join_tool_names(tool_names),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

    def collect_memory(self, tool_names: List[str]) -> Dict[str, str]:
        """Sync wrapper for summarize + retrieve.

        Prefer ``collect_memory_async`` inside an existing event loop.
        """
        if not tool_names:
            return {}

        names = self._join_tool_names(tool_names)
        try:
            summary = self.summary_tool_memory(names)
            if not summary.get("success"):
                logger.warning("summary_tool_memory failed for %s", names)
        except Exception as exc:  # noqa: BLE001
            logger.warning("summary_tool_memory error for %s: %s", names, exc)

        try:
            retrieved = self.retrieve_tool_memory(names)
        except Exception as exc:  # noqa: BLE001
            logger.warning("retrieve_tool_memory error for %s: %s", names, exc)
            return {}

        if not retrieved.get("success"):
            logger.warning("retrieve_tool_memory failed for %s", names)
            return {}

        return self._content_by_tool(retrieved)

    def get_memory_content(
        self,
        tool_names: List[str] | str,
    ) -> Optional[str]:
        """Retrieve and join memory contents for the given tools."""
        payload = self.retrieve_tool_memory(tool_names)
        if not payload.get("success"):
            return None
        contents = [content for content in self._content_by_tool(payload).values() if content]
        return "\n\n".join(contents) if contents else None
