"""Financial observation analysis using the shared round protocol."""

from .base import BaseProactiveAgentStep
from ....core.registry import R


@R.register("finance_agent_step")
class FinanceAgentStep(BaseProactiveAgentStep):
    """Read source-validated quotes and news under the user's TASK.md."""
