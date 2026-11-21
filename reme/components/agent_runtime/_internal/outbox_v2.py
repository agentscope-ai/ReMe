"""Reminders and their delivery state in v2 daily records."""

import re

import yaml

from ....schema.agent_runtime import AgentReminder
from ....schema.proactive_agent import ProactiveRecordDecision
from ....utils.proactive_record import record_decision, round_paths, round_time, split_record

_DAILY_NAME = re.compile(r"[\w-]+_[0-9]{6}(?:_[0-9]+)?\.md")
_STATE_KEY = "reminder_state"


class DailyReminderFiles:
    """Project canonical reminders from daily notes and merge state into frontmatter."""

    def __init__(self, files, directory):
        self.files = files
        self.directory = directory

    def paths(self, agent):
        """Discover dated records for the sender."""
        return round_paths(self.directory, agent.agent_name)

    def contains(self, path):
        """Identify round-note paths for dispatch by the shared outbox."""
        return path.parent.parent == self.directory and _DAILY_NAME.fullmatch(path.name) is not None

    def owns(self, path, agent):
        """Validate the sender's timestamped destination."""
        return path.parent.parent == self.directory and round_time(path, agent.agent_name) is not None

    @staticmethod
    def _decision(metadata, body):
        values = {key: metadata[key] for key in ProactiveRecordDecision.model_fields if key in metadata}
        return record_decision(values, body)

    def read(self, path):
        """The presence of runtime state distinguishes canonical reminders from drafts."""
        metadata, body = split_record(self.files.read(path))
        if _STATE_KEY not in metadata:
            return None
        decision = self._decision(metadata, body)
        if not decision.remind:
            return None
        return AgentReminder(
            priority=decision.reminder_priority,
            message=decision.reminder_message,
            paths=[path.relative_to(self.files.workspace).as_posix()],
            **metadata[_STATE_KEY],
        )

    def serialize(self, path, item):
        """Update only runtime state, preserving the note and reminder authored by the agent."""
        relative = path.relative_to(self.files.workspace).as_posix()
        if item.paths != [relative]:
            raise ValueError("a v2 reminder must reference its daily note")
        metadata, body = split_record(self.files.read(path))
        decision = self._decision(metadata, body)
        if (
            not decision.remind
            or decision.reminder_priority != item.priority
            or decision.reminder_message != item.message
        ):
            raise ValueError(f"reminder decision changed: {path}")
        metadata[_STATE_KEY] = item.model_dump(mode="json", exclude={"message", "priority", "paths"})
        return f"---\n{yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)}---{body}"
