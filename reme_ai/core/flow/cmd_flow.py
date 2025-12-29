from .base_flow import BaseFlow
from ..utils import parse_flow_expression


class CmdFlow(BaseFlow):

    def __init__(self, flow: str = "", **kwargs):
        super().__init__(**kwargs)
        self.flow = flow
        assert flow, "add `flow=<op_flow>` in cmd!"

    def _build_flow(self):
        return parse_flow_expression(self.flow)
