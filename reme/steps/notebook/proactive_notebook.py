"""Notebook updates using the shared Proactive Agent round."""

from ..proactive.agent.proactive import ProactiveAgentStep
from ...core.registry import R


@R.register("proactive_notebook_step")
class ProactiveNotebookStep(ProactiveAgentStep):
    """Update notebook records and hand remaining actions to Remy."""

    def session_options(self, agent):
        options = super().session_options(agent)
        options["config"]["skills.include_instructions"] = True
        options["injected_job_kwargs"].update(
            _automatic_notebook=True,
            _allowed_paths=self.context.get("_allowed_paths"),
        )
        options["skills"] = ["manage-todos-ideas"]
        options["job_tools"] = ["todo", "idea"]
        return options
