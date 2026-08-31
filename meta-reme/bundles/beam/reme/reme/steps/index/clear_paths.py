"""Delete stale workspace outputs before a job rebuilds them.

A tiny, generic housekeeping step: given workspace-relative ``paths`` (and/or
``config_keys`` resolved against the app config, e.g. ``daily_dir``), it removes
each target — file or directory — so the following steps start from a clean
slate. Missing targets are ignored, and anything resolving outside the workspace
is refused, so a misconfigured path can never wipe unrelated files.
"""

import shutil
from pathlib import Path

from ..base_step import BaseStep
from ...components import R


@R.register("clear_paths_step")
class ClearPathsStep(BaseStep):
    """Remove configured files/dirs under the workspace before a rebuild."""

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
        root = self.workspace_path.resolve()
        removed: list[str] = []

        for rel in self._targets():
            target = (self.workspace_path / rel).resolve()
            # Refuse anything outside the workspace, or the workspace root itself.
            if target == root or root not in target.parents:
                self.logger.warning(f"[{self.name}] refusing to clear out-of-workspace path: {rel!r}")
                continue
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
                removed.append(rel)
            elif target.exists() or target.is_symlink():
                Path(target).unlink(missing_ok=True)
                removed.append(rel)

        self.context.response.metadata["cleared_paths"] = removed
        if removed:
            self.logger.info(f"[{self.name}] cleared {removed}")
        return self.context.response
