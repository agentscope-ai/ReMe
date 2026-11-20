"""Codex process ownership and session operations shared by transport backends."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..base_component import BaseComponent
from ..job import BackgroundJob, StreamJob
from ..service.scoped_mcp import ScopedMCP
from ...enumeration import ComponentEnum
from ...schema import SessionMessage, SessionSnapshot, SessionStatus, TokenUsage

if TYPE_CHECKING:
    from openai_codex import AsyncThread


class BaseCodexAppServer(BaseComponent):
    """Own job-tool access and the public Codex session contract."""

    component_type = ComponentEnum.CODEX_APP_SERVER

    def __init__(self, codex_home: str | Path | None = None, **kwargs):
        super().__init__(**kwargs)
        self._codex_home = codex_home
        self.jobs_mcp = ScopedMCP()

    @property
    def session_path(self) -> Path:
        """Persistent Codex state under the workspace, unless an explicit home is configured."""
        if self._codex_home:
            path = Path(self._codex_home).expanduser()
            return path if path.is_absolute() else self.workspace_path / path
        return self.workspace_path / ".codex"

    @staticmethod
    def validate_job_tools(jobs: list[Any] | None) -> None:
        """Validate the currently configured job tools."""
        if any(isinstance(job, (BackgroundJob, StreamJob)) for job in jobs or []):
            raise TypeError("Codex job_tools must be non-stream request jobs")
        if any(not getattr(job, "enabled", True) for job in jobs or []):
            raise ValueError("Codex job_tools must be enabled")

    @abstractmethod
    async def release_session(self, session_id: str) -> None:
        """Release runtime ownership while retaining native history."""

    @abstractmethod
    async def get_session_status(self, session_id: str) -> SessionStatus:
        """Inspect native activity without acquiring the generation lock."""

    @abstractmethod
    async def get_session_path(self, session_id: str) -> Path | None:
        """Locate native session storage."""

    @abstractmethod
    async def session_exists(self, session_id: str) -> bool:
        """Check resumability without generating a reply."""

    @abstractmethod
    async def read_session(self, session_id: str) -> SessionSnapshot:
        """Read native display history."""

    @asynccontextmanager
    @abstractmethod
    async def append_lease(self, session_id: str) -> AsyncIterator[dict[str, SessionMessage]]:
        """Serialize append receipts independently of generation and retain pending requests."""
        yield

    @abstractmethod
    async def inject_items(self, session_id: str, items: list[dict[str, Any]]) -> None:
        """Submit native history items without waiting for generation to finish."""

    @abstractmethod
    async def wait_idle(self, thread: AsyncThread) -> None:
        """Wait for the native operation while preserving session ownership."""

    @abstractmethod
    async def token_usage(self, thread_id: str) -> TokenUsage:
        """Recover the native cumulative usage baseline."""

    @abstractmethod
    def update_token_usage(self, thread_id: str, total: TokenUsage) -> TokenUsage:
        """Advance a thread's monotonic usage baseline."""

    @asynccontextmanager
    @abstractmethod
    async def turn(
        self,
        thread: AsyncThread,
        inputs: Any,
        *,
        output_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[Any]:
        """Own one native turn and its cancellation cleanup."""
        yield

    @abstractmethod
    async def compact(self, thread: AsyncThread) -> None:
        """Own native context compaction until completion."""

    @asynccontextmanager
    @abstractmethod
    async def thread(
        self,
        thread_id: str = "",
        *,
        fork: bool = False,
        options: dict[str, Any] | None = None,
        turn_options: dict[str, Any] | None = None,
        jobs: list[Any] | None = None,
        injected: dict[str, Any] | None = None,
        mcp_startup_timeout: float = 30,
        mcp_tool_timeout: float = 300,
        keep_empty: bool = False,
    ) -> AsyncIterator[AsyncThread]:
        """Serialize one conversation while allowing independent threads to run concurrently."""
        yield
