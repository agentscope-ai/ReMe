"""Agent-owned configuration files; directory scans are the registry."""

from datetime import datetime
import re

from .base import TaskNotFoundError
from ....constants import CHAT_AGENT_NAME
from ....utils.cron_utils import next_cron_time
from ....utils.files.workspace_files import WorkspaceFiles
from ....schema.proactive_runtime import validate_agent_name
from ....schema.proactive_agent import CreateProactiveRuntimeConfig, ProactiveRuntimeDefinition, ProactiveRuntimeConfig


class RuntimeRegistry:
    """Keep one editable config per agent and no duplicate global index."""

    def __init__(self, files: WorkspaceFiles, directory: str, logger):
        self.files = files
        self.logger = logger
        self.issues: dict[str, str] = {}
        self.directory = files.path(directory)
        if self.directory == files.workspace:
            raise ValueError("reme_agent_dir must be a workspace subdirectory")

    def directory_for(self, agent: ProactiveRuntimeDefinition):
        """Every task owns one stable directory directly under the Agent root."""
        return self.files.checked(self.directory / agent.agent_name)

    def path(self, agent_name: str):
        """Resolve chat or a retained task identity, including deleted tasks."""
        validate_agent_name(agent_name)
        if agent_name == CHAT_AGENT_NAME:
            raise ValueError("Remy is not managed by AgentManager")
        return self.directory_for(self.get(agent_name, include_deleted=True))

    def _paths(self):
        return [path for path in sorted(self.directory.glob("*/config.json")) if path.parent.name != CHAT_AGENT_NAME]

    def configured_names(self) -> set[str]:
        """Reserve every persisted name, including unavailable or deleted definitions."""
        return {path.parent.name.casefold() for path in self._paths()}

    def load(self, *, include_deleted=False) -> dict[str, ProactiveRuntimeDefinition]:
        """Load valid definitions and report damaged files individually."""
        agents, issues = {}, {}
        for path in self._paths():
            try:
                agent = ProactiveRuntimeDefinition.model_validate_json(self.files.read(path))
                if path.parent != self.directory_for(agent):
                    raise ValueError(f"agent config does not match its directory: {path}")
                agents[agent.agent_name] = agent
            except (OSError, ValueError) as exc:
                issues[path.parent.name] = str(exc)

        issues.update(self._name_issues(agents))
        for name, error in issues.items():
            if self.issues.get(name) != error:
                self.logger.error(f"Proactive Runtime {name} has an invalid definition: {error}")
        self.issues = issues
        return {
            name: agent
            for name, agent in agents.items()
            if name not in issues and (include_deleted or agent.deleted_at is None)
        }

    def get(self, agent_name, *, include_deleted=False):
        """Resolve an identity and report its configuration error directly."""
        agents = self.load(include_deleted=include_deleted)
        if agent_name in self.issues:
            raise ValueError(f"Invalid Proactive Runtime {agent_name}: {self.issues[agent_name]}")
        if agent_name not in agents:
            raise TaskNotFoundError(f"proactive agent not found: {agent_name}")
        return agents[agent_name]

    @staticmethod
    def _name_issues(agents):
        names, issues = {}, {}
        for name in agents:
            previous = names.setdefault(name.casefold(), name)
            if previous != name:
                issues[previous] = issues[name] = "proactive agent names must be unique, including case"
        for key, name in names.items():
            for pattern in (r"(.+)_\d{6}", r"(.+)_\d{6}_\d+"):
                match = re.fullmatch(pattern, key)
                if match and match[1] in names:
                    message = f"agent name conflicts with timestamped records: {name}"
                    issues[name] = issues[names[match[1]]] = message
        return issues

    def save(self, agent):
        """Publish one validated definition without rewriting other agents."""
        self.files.write_json(self.directory_for(agent) / "config.json", agent.model_dump(mode="json"))

    @staticmethod
    def _validate_schedule(config: ProactiveRuntimeConfig, now: datetime) -> None:
        """Reject schedules without a next occurrence before publishing configuration."""
        if config.agent_type == "time":
            next_cron_time(config.schedule, now)

    def prepare_create(self, config, now: datetime, defaults):
        """Validate creation or restoration before publishing any task state."""
        supplied = dict(config)
        name = validate_agent_name(supplied.get("agent_name"))
        agents = self.load(include_deleted=True)
        previous = next((v for k, v in agents.items() if k.casefold() == name.casefold()), None)
        if previous is None:
            if name.casefold() in self.configured_names():
                raise ValueError("An unavailable definition already owns this name")
            values = {**defaults, **supplied}
        else:
            if previous.agent_name != name or previous.deleted_at is None:
                raise ValueError("Task already exists")
            if supplied.get("agent_type") != previous.agent_type:
                raise ValueError("Task trigger type cannot change after creation")
            if any(supplied[key] != value for key, value in previous.wiring().items() if key in supplied):
                raise ValueError("Task Runtime wiring cannot change on restore")
            values = {
                **previous.model_dump(exclude={"created_at", "deleted_at"}),
                **supplied,
                "enabled": supplied.get("enabled", True),
            }
        config = CreateProactiveRuntimeConfig.model_validate(values)
        self._validate_schedule(config, now)
        agent = ProactiveRuntimeDefinition(
            **config.model_dump(exclude={"task_description"}),
            created_at=previous.created_at if previous else now,
        )
        agents[agent.agent_name] = agent
        if issues := self._name_issues(agents):
            raise ValueError(next(iter(issues.values())))
        return agent, previous is not None

    def update(self, agent_name, patch, now: datetime):
        """Only operational settings can be changed after creation."""
        if not patch or set(patch) - {"enabled", "schedule", "min_trigger_interval_seconds", "notify_window"}:
            raise ValueError("Update requires supported runtime settings")
        agent = self.get(agent_name)
        candidate = ProactiveRuntimeDefinition.model_validate({**agent.model_dump(), **patch})
        if candidate.schedule != agent.schedule or candidate.enabled and not agent.enabled:
            self._validate_schedule(candidate, now)
        self.save(candidate)
        return candidate

    def delete(self, agent_name, now):
        """Retain configuration and workspace history as a restorable tombstone."""
        agent = self.get(agent_name)
        agent = agent.model_copy(update={"deleted_at": now})
        self.save(agent)
        return agent
