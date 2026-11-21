"""Reminder files used by v1 tasks."""

import frontmatter

from ....schema.agent_runtime import AgentReminder


class ReminderFiles:
    """Read and write the original standalone Markdown reminders."""

    def __init__(self, runtime):
        self.files = runtime.files
        self.agents = runtime.agents

    def paths(self, agent):
        """Discover reminders in this agent's directory."""
        return sorted((self.agents.directory_for(agent) / "reminder").glob("*/*.md"))

    def owns(self, path, agent):
        """Keep standalone reminders under the sender's directory."""
        return path.parent.parent == self.agents.directory_for(agent) / "reminder" and path.suffix == ".md"

    def read(self, path):
        """Only reminders with runtime-assigned IDs enter the outbox."""
        document = frontmatter.loads(self.files.read(path))
        if "id" not in document:
            return None
        metadata = dict(document.metadata)
        priority = metadata.pop("action", None)
        return AgentReminder(priority=priority, message=document.content, **metadata)

    def serialize(self, _path, item):
        """Keep the reminder text in the legacy Markdown body."""
        metadata = item.model_dump(mode="json", exclude={"message", "priority"})
        return frontmatter.dumps(frontmatter.Post(item.message, action=item.priority, **metadata)) + "\n"
