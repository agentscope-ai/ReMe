"""Codex rollout parsing plugged into the common Harness workflow."""

from datetime import datetime
from itertools import groupby
import os
from pathlib import Path

from .base_harness_connector import BaseHarnessConnectorStep
from ._codex_dialogue import CodexDialogueIndex
from .....core.registry import R
from .....utils.files.workspace_files import WorkspaceFiles


class _CodexDialogueBase(BaseHarnessConnectorStep):
    """Supply the Codex root, normalized parser and dialogue renderer."""

    source = "codex_dialogue"

    def __init__(self, codex_home=None, **kwargs):
        super().__init__(**kwargs)
        self.codex_home = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        self.description = self.get_prompt("description")

    def _index(self):
        files = WorkspaceFiles(self.workspace_path)
        return CodexDialogueIndex(
            files,
            files.path(self.config_value("resource_dir")) / self.source,
            self.codex_home,
            self._check_stopped,
        )

    def render_dialogue(self, messages, now):
        """Render localized headings while preserving message text and evidence references."""
        role_keys = {None: "user", "commentary": "commentary", "final_answer": "final_answer"}
        sections = [self.get_prompt("title")]
        for cwd, directory_messages in groupby(messages, key=lambda row: row["cwd"]):
            sections.append(self.prompt_format("directory", cwd=cwd))
            for session_id, session_messages in groupby(directory_messages, key=lambda row: row["session_id"]):
                sections.append(self.prompt_format("session", session_id=session_id))
                for turn_id, turn_messages in groupby(session_messages, key=lambda row: row["turn_id"]):
                    sections.append(self.prompt_format("turn", turn_id=turn_id))
                    for row in turn_messages:
                        timestamp = datetime.fromisoformat(row["timestamp"]).astimezone(now.tzinfo)
                        role = self.get_prompt(role_keys[row.get("phase")])
                        sections.append(
                            self.prompt_format("message", role=role, timestamp=timestamp, message_id=row["message_id"]),
                        )
                        if row["content"]:
                            sections.append(row["content"].rstrip("\n"))
                        images = row.get("image_paths", ())
                        for filename in images:
                            number = Path(filename).stem.rsplit("-", 1)[-1]
                            sections.append(self.prompt_format("image", number=number, filename=filename))
                        if missing := row["image_attachments"] - len(images):
                            sections.append(self.prompt_format("missing_images", count=missing))
        return "\n\n".join(sections) + "\n"


@R.register("codex_dialogue_init_step")
class CodexDialogueInitStep(_CodexDialogueBase):
    """Establish source cursors without importing existing dialogue."""

    async def execute(self):
        return await self.initialize()


@R.register("codex_dialogue_step")
class CodexDialogueStep(_CodexDialogueBase):
    """Collect only new normalized messages after the persisted baseline."""
