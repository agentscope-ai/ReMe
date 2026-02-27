"""Base memory agent for handling memory operations with tool-based reasoning."""

import json
from abc import ABCMeta

from ...core.enumeration import MemoryType
from ...core.op import BaseReact
from ...core.schema import MemoryNode


class BaseMemoryAgent(BaseReact, metaclass=ABCMeta):
    """Base class for memory agents that handle memory operations with tool-based reasoning."""

    memory_type: MemoryType | None = None

    @property
    def query(self) -> str:
        """query"""
        return self.context.get("query", "")

    @property
    def messages(self) -> list:
        """messages"""
        return self.context.get("messages", [])

    @property
    def description(self) -> str:
        """description"""
        return self.context.get("description", "")

    @property
    def memory_target(self) -> str:
        """memory_target"""
        return self.context.memory_target

    @property
    def history_node(self) -> MemoryNode:
        """Returns the history node."""
        return self.context.history_node

    @property
    def author(self) -> str:
        """Returns the LLM model name as the author identifier."""
        return self.llm.model_name

    @property
    def retrieved_nodes(self) -> list[MemoryNode]:
        """Returns the retrieved nodes."""
        if "retrieved_nodes" not in self.context:
            self.context.retrieved_nodes = []
        return self.context.retrieved_nodes

    @property
    def memory_target_type_mapping(self) -> dict[str, MemoryType]:
        """Get the memory target type mapping from context."""
        memory_targets = self.context.memory_targets
        memory_target_type_mapping = self.context.service_context.memory_target_type_mapping.copy()
        if memory_targets:
            return {memory_target: memory_target_type_mapping[memory_target] for memory_target in memory_targets}
        else:
            return memory_target_type_mapping

    @property
    def meta_memory_info(self) -> str:
        """Get the meta memory info from context."""
        lines = []
        for memory_target, memory_type in self.memory_target_type_mapping.items():
            line = {
                "agent": f"Agent managing {memory_type.value} memories for {memory_target}",
                "memory_target": memory_target,
            }
            lines.append(json.dumps(line, ensure_ascii=False))
        return "\n".join(lines)
