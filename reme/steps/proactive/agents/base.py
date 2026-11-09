"""Shared workspace, memory and runtime access for Remy's agents."""

import asyncio
from datetime import datetime
from pathlib import Path

from ..base import BasePaStep
from ...file_io._file_io import write_file_safe
from ...file_io._path import resolve_path
from ....components.agent_wrapper import CodexAgentWrapper
from ....utils.file_utils import get_path_lock


class BasePaAgentStep(BasePaStep):
    """Share Remy's workspace context and application-owned session state."""

    def _now(self) -> datetime:
        return self.pa.now()

    def _agent_options(self, *, injected: dict, **details) -> dict:
        """Build one consistent persona, tool binding and model override."""
        options = {
            "system_prompt": self._system_prompt(**details),
            "job_tools": list(dict.fromkeys(self.kwargs.get("job_tools") or [])),
            "injected_job_kwargs": injected,
        }
        if model := self.kwargs.get("model"):
            options["model"] = model
        return options

    async def _within_active_hours(self, operation, *, task: bool = False):
        """Apply the same deadline to session preparation, compaction and model work."""
        config = await self.pa.get_proactive_config()
        now = self._now()
        follows = not task or config.tasks_follow_active_hours
        if follows and not config.is_active(now):
            return None
        timeout = (config.active_window(now)[1] - now).total_seconds() if follows else None
        deadline = asyncio.timeout(timeout)
        try:
            async with deadline:
                return await operation()
        except TimeoutError:
            if not deadline.expired():
                raise
            return None

    def _path(self, path: str) -> Path:
        target, error = resolve_path(self.workspace_path, path)
        if error:
            raise ValueError(error)
        return target

    async def _session_exists(self, session_id: str | None) -> bool:
        """Recreate absent native state without masking transport or provider failures."""
        return session_id is not None and await self.agent_wrapper.session_exists(session_id)

    def _memory_path(self) -> Path:
        return self._path(f"{self.config_value('daily_dir')}/personal/MEMORY.md")

    async def _ensure_memory(self) -> Path:
        target = self._memory_path()
        async with await get_path_lock(target):
            if not target.exists():
                await write_file_safe(target, self.get_prompt("default_memory") + "\n")
            elif not target.is_file():
                raise ValueError("Personal memory must be a regular file")
        return target

    def _system_prompt(self, **details) -> str:
        return self.prompt_format(
            "system_prompt",
            workspace_dir=self.workspace_path,
            memory_path=self._memory_path(),
            proactive_config_path=self.pa.proactive_config_path,
            cron_path=self.pa.cron_path,
            daily_dir=self.config_value("daily_dir"),
            **details,
        )

    def _validate(self, *parameters: str) -> None:
        super()._validate(*parameters)
        if self.agent_wrapper is None:
            raise RuntimeError("PA agents require an agent_wrapper")
        if not isinstance(self.agent_wrapper, CodexAgentWrapper):
            raise NotImplementedError("PA currently supports CodexAgentWrapper only")
        if self.pa.agent_wrapper is not self.agent_wrapper:
            raise ValueError("PA jobs must use the pa component's configured agent_wrapper")
