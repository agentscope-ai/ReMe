"""Expression-based flow implementation driven by configuration objects."""

from .base_flow import BaseFlow
from ..op import BaseOp
from ..schema import FlowConfig, ToolCall


class ExpressionFlow(BaseFlow):
    """A flow implementation that constructs operations from a FlowConfig definition."""

    def __init__(self, flow_config: FlowConfig):
        """Initialize the flow using settings and metadata from a FlowConfig instance."""
        self.flow_config: FlowConfig = flow_config
        super().__init__(
            name=flow_config.name,
            stream=self.flow_config.stream,
            raise_exception=self.flow_config.raise_exception,
            enable_cache=self.flow_config.enable_cache,
            cache_path=self.flow_config.cache_path,
            cache_expire_hours=self.flow_config.cache_expire_hours,
            **flow_config.model_extra,
        )

    def _build_flow(self) -> BaseOp:
        """Generate the operation chain by parsing the flow content string."""
        return self.parse_expression(self.flow_config.flow_content)

    def _build_tool_call(self) -> ToolCall:
        """Construct a tool call representation based on configuration parameters."""
        return ToolCall(**{"description": self.flow_config.description, "parameters": self.flow_config.parameters})
