"""Agent-owned configuration files; directory scans are the registry."""

from datetime import datetime
from pathlib import Path
import re

from .files import AgentFiles
from ....schema.agent_runtime import validate_agent_id
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
        return self.files.checked(parent / agent.agent_id)

    def path(self, agent_id: str):
        """Resolve main or a retained task identity, including deleted tasks."""
        validate_agent_id(agent_id)
        if agent_id == "main":
            return self.files.checked(self.directory / "main")
        return self.directory_for(self.get(agent_id, include_deleted=True))

    def load(self, *, include_deleted=False) -> dict[str, ProactiveAgent]:
        """Validate every config, including directory identity and global uniqueness."""
        agents = []
        for pattern in ("*/config.json", "tmp/*/config.json"):
            for path in sorted(self.directory.glob(pattern)):
                if path.parent == self.directory / "main":
                    continue
                agent = ProactiveAgent.model_validate_json(self.files.read(path))
                if path.parent != self.directory_for(agent):
                    raise ValueError(f"agent config does not match its directory: {path}")
                agents.append(agent)
        self._validate_names(agents)
        return {agent.agent_id: agent for agent in agents if include_deleted or agent.deleted_at is None}

    def get(self, agent_id, *, include_deleted=False):
        """Resolve a missing identity explicitly rather than exposing dictionary errors."""
        agents = self.load(include_deleted=include_deleted)
        if agent_id not in agents:
            raise ValueError(f"proactive agent not found: {agent_id}")
        return agents[agent_id]

    @staticmethod
    def _validate_names(agents):
        names = {agent.agent_id.casefold() for agent in agents}
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

    def initialize_memory(self, directory: Path, description: str, *, language=""):
        """Initialize all Agent memories once, preserving existing files verbatim."""
        headings = (
            ("当前任务", "已确认背景", "长期结论")
            if language == "zh"
            else ("Current task", "Confirmed background", "Long-term findings")
        )
        memory = f"# {headings[0]}\n\n{description}\n\n# {headings[1]}\n\n# {headings[2]}\n"
        self.files.write(directory / "MEMORY.md", memory, create_only=True)

    def create(self, config, now: datetime, *, language=""):
        """Publish initial memory before configuration; retained memory survives restoration."""
        config = CreateProactiveAgentConfig.model_validate(config)
        agents = self.load(include_deleted=True)
        previous = next((agent for key, agent in agents.items() if key.casefold() == config.name.casefold()), None)
        if previous is not None:
            if previous.agent_id != config.name or previous.deleted_at is None:
                raise ValueError(f"proactive agent name already exists: {config.name}")
            if previous.temporary != config.temporary:
                raise ValueError("temporary is immutable for an existing agent")
        agent = ProactiveAgent(
            **config.model_dump(exclude={"description"}),
            created_at=previous.created_at if previous else now,
        )
        agents[agent.agent_id] = agent
        self._validate_names(list(agents.values()))
        self.initialize_memory(self.directory_for(agent), config.description, language=language)
        self.save(agent)
        return agent

    def update(self, agent_id, patch):
        """Update execution settings while keeping task text in its memory file."""
        agent = self.get(agent_id)
        for key, expected in (("name", agent.agent_id), ("temporary", agent.temporary)):
            if key in patch and patch[key] != expected:
                raise ValueError(f"{key} is immutable")
        values = {key: getattr(agent, key) for key in ProactiveAgentConfig.model_fields}
        config = ProactiveAgentConfig.model_validate({**values, **patch})
        agent = agent.model_copy(update=config.model_dump())
        self.save(agent)
        return agent

    def delete(self, agent_id, now):
        """Retain configuration and workspace history as a restorable tombstone."""
        agent = self.get(agent_id)
        agent = agent.model_copy(update={"deleted_at": now})
        self.save(agent)
        return agent
