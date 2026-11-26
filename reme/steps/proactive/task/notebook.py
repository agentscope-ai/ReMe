"""Evidence-backed follow-up of existing Todo and Idea records."""

from .base import BaseProactiveAgentStep
from ....core.registry import R


@R.register("notebook_agent_step")
class NotebookAgentStep(BaseProactiveAgentStep):
    """Specialize skills and trusted tool permissions, not the round executor."""

    skill_names = ("manage-notebook",)

    def session_options(self, agent):
        """Build the formal role binding independently of temporary fork tools."""
        options = super().session_options(agent)
        options["sandbox"] = "read-only"
        options["job_tools"] = ["manage_notebook"]
        options["injected_job_kwargs"]["_automatic_notebook"] = True
        return options

    def skill_templates(self):
        """Select the dynamic instructions added by this domain workflow."""
        return [*super().skill_templates(), ("notebook_skill", "notebook-follow-up", "notebook_skill_description")]
