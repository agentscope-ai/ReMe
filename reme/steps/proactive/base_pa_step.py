"""Shared workspace, memory and runtime access for Remy's agents."""

import asyncio
from datetime import datetime
from pathlib import Path

from ..base_step import BaseStep, Ref
from ..file_io._file_io import write_file_safe
from ..file_io._path import resolve_path
from ...components import PaRuntime
from ...components.agent_wrapper import CodexAgentWrapper
from ...utils.file_utils import get_path_lock


class BasePaStep(BaseStep):
    """Share Remy's workspace context and application-owned session state."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    def _now(self) -> datetime:
        return self.pa._now()  # pylint: disable=protected-access

    async def _reply(self, query: str, *, task: bool = False, immediate: bool = False, **kwargs):
        """Return None when active hours end; cancel native work at the deadline."""
        if immediate:
            return await self.agent_wrapper.reply(query, **kwargs)
        config = await self.pa.get_proactive_config()
        now = self._now()
        follows = not task or config.tasks_follow_active_hours
        if follows and not config.is_active(now):
            return None
        timeout = (config.active_window(now)[1] - now).total_seconds() if follows else None
        deadline = asyncio.timeout(timeout)
        try:
            async with deadline:
                return await self.agent_wrapper.reply(query, **kwargs)
        except TimeoutError:
            if not deadline.expired():
                raise
            return None

    def _path(self, path: str) -> Path:
        target, error = resolve_path(self.workspace_path, path)
        if error:
            raise ValueError(error)
        return target

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

    def _handoff_prompt(self, name: str, **details) -> str:
        return self.prompt_format(name, handoff_rules=self.get_prompt("handoff_rules"), **details)

    def _validate(self, *parameters: str) -> None:
        if self.app_context is None or self.agent_wrapper is None:
            raise RuntimeError("PA jobs require an ApplicationContext and agent_wrapper")
        if not isinstance(self.agent_wrapper, CodexAgentWrapper):
            raise NotImplementedError("PA currently supports CodexAgentWrapper only")
        if self.pa.agent_wrapper is not self.agent_wrapper:
            raise ValueError("PA jobs must use the pa component's configured agent_wrapper")
        unexpected = self.context.data.keys() - {*parameters, "metadata"}
        if unexpected:
            raise ValueError(f"Unexpected PA parameters: {', '.join(sorted(unexpected))}")
