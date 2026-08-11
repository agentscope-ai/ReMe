"""Process-local runtime configuration shared by Meta-ReMe tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetaReMeToolRuntime:
    """Runtime values established by ``meta-reme/run.py`` for agent tools."""

    workspace: Path | None = None
    validation_concurrency: int | None = None

    def configure(self, workspace: Path, validation_concurrency: int) -> None:
        """Replace the active run values while preserving singleton identity."""

        if validation_concurrency < 1:
            raise ValueError("validation_concurrency must be at least 1")
        self.workspace = Path(workspace).resolve()
        self.validation_concurrency = validation_concurrency

    def require_configured(self) -> tuple[Path, int]:
        """Return configured values or fail clearly outside the main workflow."""

        if self.workspace is None or self.validation_concurrency is None:
            raise RuntimeError("Meta-ReMe tool runtime is not configured; run meta-reme/run.py first")
        return self.workspace, self.validation_concurrency


TOOL_RUNTIME = MetaReMeToolRuntime()
