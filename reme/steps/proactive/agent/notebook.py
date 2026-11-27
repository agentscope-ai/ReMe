"""Evidence-backed follow-up of existing Todo and Idea records."""

from .base import BaseProactiveAgentStep
from ....core.registry import R


@R.register("notebook_agent_step")
class NotebookAgentStep(BaseProactiveAgentStep):
    """Bind Notebook tools and skills to the shared round executor."""

    skill_names = (*BaseProactiveAgentStep.skill_names, "manage-notebook")
    job_tools = ("manage_notebook",)

    def session_options(self, agent, now):
        """Build the formal role binding independently of temporary fork tools."""
        options = super().session_options(agent, now)
        options["injected_job_kwargs"]["_automatic_notebook"] = True
        return options
