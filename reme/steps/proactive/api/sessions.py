"""Inspect or select Chat Agent without depending on agent prompt preparation."""

from ..base import BasePaStep
from ....components import R


@R.register("pa_session_step")
class PaSessionStep(BasePaStep):
    """Read selection and activity without waiting behind generation."""

    async def execute(self):
        self._validate()
        self.context.response.answer, error = await self.pa.sessions.get_selected_session()
        if error is not None:
            self.context.response.metadata["error"] = error
        return self.context.response


@R.register("pa_select_session_step")
class PaSelectSessionStep(BasePaStep):
    """Select a source conversation while no Chat Agent operation is running."""

    async def execute(self):
        self._validate("session_id")
        identifier = self.context.get("session_id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("session_id is required")
        await self.pa.sessions.select_existing_session(identifier)
        self.context.response.answer = {"session_id": identifier}
        return self.context.response
