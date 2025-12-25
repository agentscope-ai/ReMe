from abc import ABC
from pathlib import Path

from flowllm.core.schema import ToolCall
from flowllm.core.storage import CacheHandler

from .. import C, BaseAsyncToolOp
from ..enumeration import MemoryType


@C.register_op()
class BaseMemoryToolOp(BaseAsyncToolOp, ABC):
    """
    Base class for memory tool operations.
    
    This abstract base class provides common functionality for memory-related tools,
    including configuration handling, workspace identification, and tool call construction.
    """

    def __init__(self,
                 enable_multiple: bool = True,
                 enable_thinking_params: bool = False,
                 memory_metadata_dir: str = "./memory_metadata",
                 **kwargs):
        """
        Initialize the BaseMemoryToolOp.
        
        Args:
            enable_multiple (bool): Whether to enable multiple item operations. Defaults to True.
            enable_thinking_params (bool): Whether to include thinking parameters in the schema. Defaults to False.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.enable_multiple: bool = enable_multiple
        self.enable_thinking_params: bool = enable_thinking_params
        self.memory_metadata_dir: Path = Path(memory_metadata_dir)
        self._metadata_handler: CacheHandler | None = None

    def build_input_schema(self) -> dict:
        """
        Build the input schema for single item operations.
        
        This method should be overridden by subclasses to define
        the specific input schema for single item operations.
        
        Returns:
            dict: The input schema definition.
        """
        return {}

    def build_multiple_input_schema(self) -> dict:
        """
        Build the input schema for multiple item operations.
        
        This method should be overridden by subclasses to define
        the specific input schema for multiple item operations.
        
        Returns:
            dict: The input schema definition for multiple items.
        """
        return {}

    def build_tool_call(self) -> ToolCall:
        """
        Build and return a ToolCall object with appropriate schema and description.
        
        Constructs a ToolCall with either single or multiple item input schema
        based on the enable_multiple flag, and optionally adds thinking parameters.
        
        Returns:
            ToolCall: Configured ToolCall object.
        """
        if self.enable_multiple:
            input_schema = self.build_multiple_input_schema()
        else:
            input_schema = self.build_input_schema()

        if self.enable_thinking_params and "thinking" not in input_schema:
            input_schema = {
                "thinking": {
                    "type": "string",
                    "description": "Your thinking and reasoning about how to fill in the parameters",
                    "required": True,
                },
                **input_schema,
            }

        tool_name: str = "tool" + ("_multiple" if self.enable_multiple else "")
        return ToolCall(
            **{
                "description": self.get_prompt(tool_name),
                "input_schema": input_schema,
            }
        )

    @property
    def metadata_handler(self):
        if self._metadata_handler is None:
            self._metadata_handler = CacheHandler(self.memory_metadata_dir / self.workspace_id)
        return self._metadata_handler

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType(self.context.get("memory_type"))

    @property
    def memory_target(self) -> str:
        return self.context.get("memory_target", "")

    @property
    def ref_memory_id(self) -> str:
        return self.context.get("ref_memory_id", "")

    @property
    def author(self) -> str:
        return self.context.get("author", "")

    @property
    def workspace_id(self) -> str:
        return self.context.get("workspace_id", "default")