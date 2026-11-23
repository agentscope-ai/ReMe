"""Notebook updates using the shared Proactive Agent round."""

from pathlib import PurePosixPath
from textwrap import indent

from pydantic import Field, model_validator

from ..proactive.agent.proactive import ProactiveAgentStep
from ...core.registry import R
from ...schema.proactive_agent import ProactiveRecordDecision


class _NotebookRecord(ProactiveRecordDecision):
    """The shared Markdown record content returned by the read-only session."""

    body: str = Field(max_length=100000)

    @model_validator(mode="after")
    def require_reminder_body(self):
        """Give Remy a source record when work needs to be handed off."""
        if self.reminder_priority != "skip" and not self.body.strip():
            raise ValueError("A reminder requires a Markdown body")
        return self


@R.register("proactive_notebook_step")
class ProactiveNotebookStep(ProactiveAgentStep):
    """Update notebook records and hand remaining actions to Remy."""

    def session_options(self, agent, events):
        options = super().session_options(agent, events)
        options["config"]["skills.include_instructions"] = True
        options["injected_job_kwargs"].update(
            _proactive_event_ids=[event["id"] for event in events],
            _allowed_paths=self.context.get("_allowed_paths"),
        )
        return {
            **options,
            "skills": ["manage-todos-ideas"],
            "job_tools": ["todo", "idea"],
            "sandbox": "read-only",
            "output_schema": _NotebookRecord.model_json_schema(),
        }

    def _query(self, events, paths, now, task_content, *, is_cron):
        inputs = [self._inline_codex_dialogue(event) for event in events]
        return super()._query(inputs, paths, now, task_content, is_cron=is_cron)

    def _inline_codex_dialogue(self, event):
        """Put the published text in the prompt and keep image paths available."""
        if event["source"] != "codex_dialogue":
            return event
        source = next((path for path in event["paths"] if PurePosixPath(path).name == "dialogue.md"), None)
        if source is None:
            return event
        files = self.proactive_manager.files
        path = files.path(source)
        self.require_paths(path)
        dialogue = self.prompt_format(
            "codex_dialogue_input",
            source=source,
            dialogue=indent(files.read(path).rstrip(), "> ", lambda _line: True),
        )
        return {
            **event,
            "description": f"{event['description']}\n\n{dialogue}",
            "paths": [attachment for attachment in event["paths"] if attachment != source],
        }

    def _validate_output(self, record, reply, *, is_cron, repairing):
        if reply is None:
            raise ValueError("A successful reply is required to save notebook output")
        output = _NotebookRecord.model_validate(reply["structured_output"])
        if output.reminder_priority == "skip" and not is_cron:
            record.check_version(record.path)
            return None
        record.create(output.model_dump(exclude={"body"}), output.body)
        return self._read_record(record.path, None, is_cron=is_cron)
