"""Command-based flow implementation for parsing and executing operation sequences."""

from .base_flow import BaseFlow
from ..op import BaseOp


class CmdFlow(BaseFlow):
    """A flow class that builds an operation chain from a string expression."""

    def __init__(self, flow: str = "", **kwargs):
        """Initialize the command flow with a string-based operation definition."""
        super().__init__(**kwargs)
        self.flow = flow
        assert flow, "add `flow=<op_flow>` in cmd!"

    def _build_flow(self) -> BaseOp:
        """Parse the stored flow expression into a functional operation object."""
        return self.parse_expression(self.flow)
