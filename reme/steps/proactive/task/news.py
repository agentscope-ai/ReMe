"""Scheduled news preparation using the selected wrapper's search capability."""

from .base import BaseProactiveAgentStep
from ....core.registry import R


@R.register("news_agent_step")
class NewsAgentStep(BaseProactiveAgentStep):
    """Verify original sources before preparing the briefing record."""
