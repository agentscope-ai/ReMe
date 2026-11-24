"""Read reminder content from daily records and state from the task outbox."""

from pathlib import Path

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from ....schema.proactive_agent import ProactiveRecordDecision
from ....schema.proactive_runtime import ChatMessageId
from ....utils.proactive_record import is_date_directory, is_round_record, split_record


class ReminderState(BaseModel):
    """The delivery and processing state owned by ReMe."""

    model_config = ConfigDict(extra="forbid")

    delivered_at: AwareDatetime
    read_at: AwareDatetime | None = None
    chat_message_ids: list[ChatMessageId] = Field(default_factory=list)


class ReminderFiles:
    """Use one outbox state file for each registered daily record."""

    def __init__(self, files, daily_dir, agent_dir):
        self.files = files
        self.daily_dir = files.path(daily_dir)
        self.relative_dir = self.daily_dir.relative_to(files.workspace)
        self.agent_dir = agent_dir

    def _outbox(self, agent_name):
        return self.files.checked(self.agent_dir / agent_name / "outbox" / self.relative_dir)

    def state_path(self, agent_name, path):
        """Locate the registration and Chat state for one source file."""
        return self.files.checked(self._outbox(agent_name) / path.parent.name / path.with_suffix(".json").name)

    def paths(self, agent_name):
        """Discover sources from the task's persisted registrations."""
        for state in self._outbox(agent_name).glob("????-??-??/*.json"):
            path = self.daily_dir / state.parent.name / state.with_suffix(".md").name
            if self.owns(path.relative_to(self.files.workspace).as_posix(), agent_name):
                yield path, state

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

    def decision(self, path):
        """Read the model's reminder decision from the original Markdown."""
        metadata, _ = split_record(self.files.read(path))
        return ProactiveRecordDecision.parse(metadata)

    def read(self, agent_name, path):
        """Join framework-owned state with the current reminder text."""
        try:
            state = ReminderState.model_validate_json(self.files.read(self.state_path(agent_name, path)))
        except FileNotFoundError:
            return None
        ref = path.relative_to(self.files.workspace).as_posix()
        item = self.decision(path).reminder(ref)
        if item is None:
            raise ValueError(f"Registered record no longer requests a reminder: {path}")
        return item.model_copy(update=state.model_dump())

    def write(self, agent_name, path, item):
        """Persist delivery state without rewriting the original Markdown."""
        state = item.model_dump(mode="json", include=set(ReminderState.model_fields), exclude_defaults=True)
        self.files.write_json(self.state_path(agent_name, path), state)
