"""Deliver pending reminders independently of proactive reasoning rounds."""

from ._lifecycle import require_stop_event, wait_for_stop

from ..base import BasePaStep
from ....components import R


@R.register("reminder_delivery_step")
class ReminderDeliveryStep(BasePaStep):
    """Use the background Job lifecycle for delivery and ten-second idle checks."""

    async def execute(self):
        stop = require_stop_event(self.context.stop_event)
        while not stop.is_set():
            try:
                await self.pa.deliver_pending_reminder()
            except Exception:
                self.logger.exception("Pending reminder delivery failed")
            await wait_for_stop(stop, 0.5)
        return self.context.response
