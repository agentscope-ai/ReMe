"""Agent-owned configuration files; directory scans are the registry."""

from datetime import datetime
from pathlib import Path
import re

from .files import AgentFiles
from ....schema.agent_runtime import validate_agent_name
from ....schema.proactive_agent import CreateProactiveAgentConfig, ProactiveAgent, ProactiveAgentConfig


class AgentRegistry:
    """Keep one editable config per agent and no duplicate global index."""

    def __init__(self, files: AgentFiles, directory: str):
        self.files = files
        self.directory = files.path(directory)
        if self.directory == files.workspace:
            raise ValueError("reme_agent_dir must be a workspace subdirectory")

    def directory_for(self, agent: ProactiveAgent):
        """Temporary status determines a stable location for the identity."""
        parent = self.directory / "tmp" if agent.temporary else self.directory
        return self.files.checked(parent / agent.agent_name)

    def path(self, agent_name: str):
        """Resolve chat or a retained task identity, including deleted tasks."""
        validate_agent_name(agent_name)
        if agent_name == "Remy":
            return self.files.checked(self.directory / "Remy")
        return self.directory_for(self.get(agent_name, include_deleted=True))

    def load(self, *, include_deleted=False) -> dict[str, ProactiveAgent]:
        """Validate every config, including directory identity and global uniqueness."""
        agents = []
        for pattern in ("*/config.json", "tmp/*/config.json"):
            for path in sorted(self.directory.glob(pattern)):
                if path.parent == self.directory / "Remy":
                    continue
                agent = ProactiveAgent.model_validate_json(self.files.read(path))
                if path.parent != self.directory_for(agent):
                    raise ValueError(f"agent config does not match its directory: {path}")
                agents.append(agent)
        self._validate_names(agents)
        return {agent.agent_name: agent for agent in agents if include_deleted or agent.deleted_at is None}

    def get(self, agent_name, *, include_deleted=False):
        """Resolve a missing identity explicitly rather than exposing dictionary errors."""
        agents = self.load(include_deleted=include_deleted)
        if agent_name not in agents:
            raise ValueError(f"proactive agent not found: {agent_name}")
        return agents[agent_name]

    @staticmethod
    def _validate_names(agents):
        names = {agent.agent_name.casefold() for agent in agents}
        if len(names) != len(agents):
            raise ValueError("proactive agent names must be unique, including case")
        for name in names:
            for pattern in (r"(.+)_\d{6}", r"(.+)_\d{6}_\d+"):
                match = re.fullmatch(pattern, name)
                if match and match[1] in names:
                    raise ValueError(f"agent name conflicts with timestamped records: {name}")

    def save(self, agent):
        """Publish one validated definition without rewriting other agents."""
        self.files.write_json(self.directory_for(agent) / "config.json", agent.model_dump(mode="json"))

    def initialize_memory(self, directory: Path, task_memory: str):
        """Write initial text verbatim once, preserving existing Agent memories."""
        self.files.write(directory / "MEMORY.md", task_memory, create_only=True)

    def create(self, config, now: datetime):
        """Publish the task and report restoration from the same registry snapshot."""
        config = CreateProactiveAgentConfig.model_validate(config)
        agents = self.load(include_deleted=True)
        previous = next(
            (agent for key, agent in agents.items() if key.casefold() == config.agent_name.casefold()),
            None,
        )
        if previous is not None:
            if previous.agent_name != config.agent_name or previous.deleted_at is None:
                raise ValueError(f"proactive agent name already exists: {config.agent_name}")
            if previous.temporary != config.temporary:
                raise ValueError("temporary is immutable for an existing agent")
        agent = ProactiveAgent(
            **config.model_dump(exclude={"task_memory"}),
            created_at=previous.created_at if previous else now,
        )
        agents[agent.agent_name] = agent
        self._validate_names(list(agents.values()))
        self.initialize_memory(self.directory_for(agent), config.task_memory)
        self.save(agent)
        return agent, previous is not None

    def update(self, agent_name, patch):
        """Update execution settings while keeping task text in its memory file."""
        agent = self.get(agent_name)
        for key, expected in (("agent_name", agent.agent_name), ("temporary", agent.temporary)):
            if key in patch and patch[key] != expected:
                raise ValueError(f"{key} is immutable")
        values = {key: getattr(agent, key) for key in ProactiveAgentConfig.model_fields}
        config = ProactiveAgentConfig.model_validate({**values, **patch})
        agent = ProactiveAgent.model_validate({**agent.model_dump(), **config.model_dump()})
        self.save(agent)
        return agent

    def delete(self, agent_name, now):
        """Retain configuration and workspace history as a restorable tombstone."""
        agent = self.get(agent_name)
        agent = agent.model_copy(update={"deleted_at": now})
        self.save(agent)
        return agent
