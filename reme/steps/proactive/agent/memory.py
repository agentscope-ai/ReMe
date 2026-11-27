"""Project and personal-memory follow-up."""

from .base import BaseProactiveAgentStep
from ....core.registry import R


@R.register("memory_agent_step")
class MemoryAgentStep(BaseProactiveAgentStep):
    """Use the shared round with project-follow-up requirements."""
