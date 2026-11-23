"""Read and update reminders in daily records."""

from pathlib import Path

import yaml

from ....schema.proactive_runtime import AgentReminder
from ....utils.proactive_record import is_date_directory, is_round_record, round_paths, split_record


class ReminderFiles:
    """Keep the reminder and its runtime state in the source record."""

    def __init__(self, files, daily_dir):
        self.files = files
        self.daily_dir = files.path(daily_dir)
        self.relative_dir = self.daily_dir.relative_to(files.workspace)

    def paths(self, agent_name):
        """Find the task's dated records."""
        return round_paths(self.daily_dir, agent_name)

    def owns(self, ref, agent_name):
        """Recognize a canonical reference even if its record is temporarily unavailable."""
        if not isinstance(ref, str):
            return False
        path = Path(ref)
        return (
            path.parent.parent == self.relative_dir
            and path.as_posix() == ref
            and is_date_directory(path.parent)
            and is_round_record(path, agent_name)
        )

    def read(self, path, *, draft=False):
        """Read a delivered reminder or a candidate from an input receipt."""
        metadata, _ = split_record(self.files.read(path))
        if not draft and "delivered_at" not in metadata:
            return None
        item = AgentReminder(
            priority=metadata.pop("reminder_priority", None),
            message=metadata.pop("reminder_message", None),
            id=path.relative_to(self.files.workspace).as_posix(),
            **metadata,
        )
        if not draft and item.delivered_at is None:
            raise ValueError("delivered_at must be a timestamp")
        return item

    def serialize(self, path, item):
        """Update frontmatter while preserving the exact Markdown body."""
        _, body = split_record(self.files.read(path))
        state = item.model_dump(mode="json", exclude={"message", "priority", "id"}, exclude_defaults=True)
        metadata = {"reminder_priority": item.priority, "reminder_message": item.message, **state}
        return f"---\n{yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)}---{body}"
