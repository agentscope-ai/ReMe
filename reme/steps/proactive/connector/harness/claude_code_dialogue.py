"""Claude Code native transcripts plugged into the common Harness workflow."""

import os
from pathlib import Path

from .base import BaseHarnessConnectorStep
from ._claude_code_index import ClaudeCodeDialogueIndex
from .....core.registry import R


@R.register("claude_code_dialogue_step")
class ClaudeCodeDialogueStep(BaseHarnessConnectorStep):
    """Collect new user prompts, visible progress, final answers and embedded images."""

    source = "claude_code_dialogue"
    index_class = ClaudeCodeDialogueIndex

    def __init__(self, claude_config_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.source_root = (
            Path(claude_config_dir or os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"
        )


@R.register("claude_code_dialogue_init_step")
class ClaudeCodeDialogueInitStep(ClaudeCodeDialogueStep):
    """Establish the first reading baseline without importing existing history."""

    async def execute(self):
        return await self.initialize()
