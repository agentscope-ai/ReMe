from .base_flow import BaseFlow
from ..schema import FlowConfig, ToolCall
from ..utils import parse_flow_expression


class ExpressionFlow(BaseFlow):

    def __init__(self, flow_config: FlowConfig):
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

    def _build_flow(self):
        return parse_flow_expression(self.flow_config.flow_content)

    def _build_tool_call(self) -> ToolCall:
        if self.flow_op.tool_call is not None:
            return self.flow_op.tool_call
        else:
            return ToolCall(description=self.flow_config.description, input_schema=self.flow_config.input_schema)
