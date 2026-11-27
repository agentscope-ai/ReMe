"""Codex rollout parsing plugged into the common Harness workflow."""

import os
from pathlib import Path

from .base_harness_connector import BaseHarnessConnectorStep
from ._codex_dialogue import CodexDialogueIndex
from .....core.registry import R


@R.register("codex_dialogue_step")
class CodexDialogueStep(BaseHarnessConnectorStep):
    """Supply the Codex root, normalized parser and dialogue renderer."""

    source = "codex_dialogue"
    index_class = CodexDialogueIndex

    def __init__(self, codex_home=None, **kwargs):
        super().__init__(**kwargs)
        self.source_root = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "sessions"


@R.register("codex_dialogue_init_step")
class CodexDialogueInitStep(CodexDialogueStep):
    """Establish source cursors without importing existing dialogue."""

    async def execute(self):
        return await self.initialize()
