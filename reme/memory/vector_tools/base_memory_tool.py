"""Base class for memory tool"""

from abc import ABCMeta
from pathlib import Path

from ...core.enumeration import MemoryType
from ...core.op import BaseTool
from ...core.schema import ToolCall, MemoryNode, ToolAttr


class BaseMemoryTool(BaseTool, metaclass=ABCMeta):
    """Base class for memory tool"""

    def __init__(
        self,
        enable_multiple: bool = True,
        enable_thinking_params: bool = False,
        profile_dir: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_multiple: bool = enable_multiple
        self.enable_thinking_params: bool = enable_thinking_params
        self.profile_dir: str = profile_dir

    def _build_tool_call(self) -> ToolCall:
        """Build and return the tool call schema"""

    def _build_multiple_tool_call(self) -> ToolCall:
        """Build and return the multiple tool call schema"""

    @property
    def tool_call(self) -> ToolCall | None:
        """Get the tool call schema."""
        if self._tool_call is None:
            if self.enable_multiple:
                self._tool_call = self._build_multiple_tool_call()
            else:
                self._tool_call = self._build_tool_call()
            self._tool_call.name = self._tool_call.name or self.name

            # Add thinking parameter if enabled
            if self.enable_thinking_params:
                parameters = self._tool_call.parameters
                if parameters and parameters.properties is not None:
                    if "thinking" not in parameters.properties:
                        parameters.properties = {
                            "thinking": ToolAttr(
                                type="string",
                                description="Your complete and detailed thinking process "
                                "about how to fill in each parameter",
                            ),
                            **parameters.properties,
                        }
                        if parameters.required is not None:
                            parameters.required = ["thinking", *parameters.required]
                        else:
                            parameters.required = ["thinking"]
        return self._tool_call

    @property
    def memory_type(self) -> MemoryType:
        """Get the memory type from context."""
        return self.memory_target_type_mapping[self.memory_target]

    @property
    def memory_target(self) -> str:
        """Get the memory target from context."""
        if "memory_target" in self.context:
            return self.context.memory_target
        elif len(self.memory_target_type_mapping) == 1:
            return list(self.memory_target_type_mapping.keys())[0]
        else:
            raise ValueError("memory_target is not specified in context or memory_target_type_mapping!")

    @property
    def history_id(self) -> str:
        """Get the history node from context."""
        if "history_node" in self.context:
            return self.context.history_node.memory_id
        return ""

    @property
    def retrieved_nodes(self) -> list[MemoryNode]:
        """Get the retrieved nodes from context."""
        return self.context.retrieved_nodes

    @property
    def author(self) -> str:
        """Get the author from context."""
        return self.context.author

    @property
    def memory_nodes(self) -> list[MemoryNode | str]:
        """Get the memory nodes from context."""
        if "memory_nodes" not in self.context:
            self.context.memory_nodes = []
        return self.context.memory_nodes

    @property
    def memory_target_type_mapping(self) -> dict[str, MemoryType]:
        """Get the memory target type mapping from context."""
        return self.context.service_context.memory_target_type_mapping

    @property
    def profile_path(self) -> Path:
        """Get the path to the profile directory for the current collection."""
        return Path(self.profile_dir) / self.vector_store.collection_name
