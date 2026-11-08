"""Set the next proactive wakeup delay."""

from ..loop import WAKEUP_KEY, _minutes
from ...base_step import BaseStep
from ....components import R


@R.register("set_next_wakeup_step")
class SetNextWakeupStep(BaseStep):
    """Set this round's next delay; the loop consumes it once."""

    async def execute(self):
        minutes = _minutes(self.context.get("minutes"))
        self.app_context.metadata[WAKEUP_KEY] = minutes
        self.context.response.answer = {"minutes": minutes}
        return self.context.response
