"""Wait for required workspace files before continuing a job."""

import asyncio
import time
from pathlib import Path

from ..base_step import BaseStep
from ...components import R


@R.register("wait_for_paths_step")
class WaitForPathsStep(BaseStep):
    """Block until configured workspace-relative paths exist."""

    def _targets(self) -> list[str]:
        """Collect workspace-relative targets from ``paths`` + ``config_keys``."""
        rels: list[str] = list(self.kwargs.get("paths") or [])
        for key in self.kwargs.get("config_keys") or []:
            value = self.config_value(key)
            if value:
                rels.append(str(value))
        return rels

    async def execute(self):
        assert self.context is not None
        poll_seconds = float(self.kwargs.get("poll_seconds", 5.0))
        log_every_seconds = float(self.kwargs.get("log_every_seconds", 60.0))
        root = self.workspace_path.resolve()
        targets: list[tuple[str, Path]] = []

        for rel in self._targets():
            target = (self.workspace_path / rel).resolve()
            if target == root or root not in target.parents:
                raise ValueError(f"wait_for_paths_step refuses out-of-workspace path: {rel!r}")
            targets.append((rel, target))

        if not targets:
            return self.context.response

        start = time.monotonic()
        last_log_at = 0.0
        while True:
            missing = [rel for rel, target in targets if not target.exists()]
            if not missing:
                waited_seconds = time.monotonic() - start
                self.context.response.metadata["waited_for_paths"] = [rel for rel, _ in targets]
                self.context.response.metadata["waited_seconds"] = waited_seconds
                self.logger.info(f"[{self.name}] required paths are ready after {waited_seconds:.1f}s")
                return self.context.response

            now = time.monotonic()
            if now - last_log_at >= log_every_seconds:
                waited_seconds = now - start
                self.logger.info(f"[{self.name}] waiting for {missing} ({waited_seconds:.1f}s elapsed)")
                last_log_at = now
            await asyncio.sleep(poll_seconds)
