"""Read the current inputs and history before writing a single daily decision."""

import yaml

from ._artifacts import RoundOutputs
from .proactive import ProactiveAgentStep
from ....core.registry import R
from ....schema.proactive_agent import ProactiveRunResult
from ....utils.proactive_record import record_decision, round_paths, split_record


@R.register("proactive_agent_step_v2")
class ProactiveAgentStepV2(ProactiveAgentStep):
    """Share turn execution with v1 and use a daily note for the whole result."""

    def __init__(self, max_recent_records=10, max_search_records=10, **kwargs):
        super().__init__(**kwargs)
        for name, value in (("max_recent_records", max_recent_records), ("max_search_records", max_search_records)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_recent_records = max_recent_records
        self.max_search_records = max_search_records

    def _paths(self, agent, now):
        runtime = self.agent_runtime
        daily = runtime.files.path(self.config_value("daily_dir"))
        day = daily / now.date().isoformat()
        stem = f"{agent.agent_name}_{now:%H%M%S}"
        record = runtime.files.checked(day / f"{stem}.md")
        collision = 0
        while record.exists():
            collision += 1
            record = runtime.files.checked(day / f"{stem}_{collision}.md")
        return {"daily_dir": daily, "task_path": runtime.agents.task_path(agent), "record_path": record}

    def _make_artifacts(self, paths):
        return RoundOutputs(self.agent_runtime.files, (paths["record_path"],))

    def _recent_records(self, paths):
        files = self.agent_runtime.files
        result = []
        for path in round_paths(paths["daily_dir"], paths["task_path"].parent.name):
            if path == paths["record_path"]:
                continue
            try:
                metadata, body = split_record(files.read(path))
            except (OSError, ValueError, RuntimeError, yaml.YAMLError):
                continue
            if not body.strip() and metadata.get("remind") is not True:
                continue
            result.append(path.relative_to(files.workspace).as_posix())
            if len(result) == self.max_recent_records:
                break
        return result

    def _query(self, events, paths, now, task_content, *, is_cron, **prompt_values):
        recent_records = "\n".join(f"- {path}" for path in self._recent_records(paths))
        return super()._query(
            events,
            paths,
            now,
            task_content,
            is_cron=is_cron,
            recent_records=recent_records,
            has_recent=bool(recent_records),
            no_recent=not recent_records,
            max_search_records=self.max_search_records,
            **prompt_values,
        )

    def _read_artifacts(self, paths, _before, *, is_cron):
        files = self.agent_runtime.files
        path = paths["record_path"]
        metadata, body = split_record(files.read(path))
        decision = record_decision(metadata, body)
        self._warn_local_links(path, body)
        if not decision.remind:
            return ProactiveRunResult(notifications=[])
        if not is_cron and not decision.has_content:
            raise ValueError("an event reminder requires a new record")
        self._warn_local_links(path, decision.reminder_message)
        return ProactiveRunResult(
            notifications=[
                {
                    "priority": decision.reminder_priority,
                    "message": decision.reminder_message,
                    "paths": [path.relative_to(files.workspace).as_posix()],
                },
            ],
        )
